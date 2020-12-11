import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tqdm.notebook import tqdm
import scipy.signal
from tensorflow.python.training.tracking.data_structures import ListWrapper


class mesoSfM:
    def __init__(self, stack, ul_coords, recon_shape, ul_offset=(0, 0), batch_size=None, scale=1, momentum=None,
                 batch_across_images=False, report_error_map=False, restrict_function='clip'):
        # stack: stack of images to be stitched; 3D tensor of shape num_images, num_rows, num_cols, num_channels;
        # ul_coords: array of upper-left coordinates of the images (in pixels); 2D tensor of shape num_images, 2 (row,
        # col; y, x);
        # recon_shape: shape of the final reconstruction/stitched image (row, col, channels);
        # ul_offset: length-2 vector (row, col) specifying a constant offset for all the ul_coords (in pixels);
        # batch_size: if None, then don't use batching;
        # scale: factor between 0 and 1 specifying how much to downsample the reconstruction;
        # momentum: only relevant for batching; specifies how much averaging of previous iterations to use for recon;
        # either both batch_size and momentum must be non-None, or both must be None;
        # batch_across_images: if you're batching (i.e., batch_size is not None), then if true, then batch across the
        # image dimension; otherwise, batch across pixels; (if batching while using unet, then you must batch across
        # image dimension);
        # report_error_map:
        # restrict_function choice decides what to do if a point goes beyond the boundaries; 'clip' or 'mod' or
        # 'mod_with_random_shifts';

        self.tf_dtype = tf.float32
        self.np_dtype = np.float32

        self.stack = np.uint8(stack)  # cast is 8-bit to save memory for the batch generator; cast batch to float32;
        self.num_channels = self.stack.shape[3]  # number of channels; stack must at least have a singleton dim 3;
        self.num_images = self.stack.shape[0]  # number of images in dataset;
        self.ul_coords = np.uint16(ul_coords)  # uint8 is too narrow, because these pixel coordinates are large;
        self.recon_shape_base = recon_shape  # this will be the base recon_shape; the effective recon_shape will depend
        # on the scale factor that the user specifies;
        self.ul_offset = np.array(ul_offset)
        self.scale = scale
        self.momentum = momentum
        self.batch_size = batch_size
        self.batch_across_images = batch_across_images
        self.report_error_map = report_error_map
        self.restrict_function = restrict_function
        self.sig_proj = .42465  # for the intepolation kernel width;
        self.subtract_min_from_height_map = True
        self.optimizer = tf.keras.optimizers.Adam

        # unet parameters if relevant; define these manually if needed;
        self.filters_list = None
        self.skip_list = None
        self.unet_scale = .01  # to scale the output of the unet
        self.output_nonlinearity = 'linear'  # 'linear' or 'leaky_relu'
        self.upsample_method = 'bilinear'  # 'bilinear' or 'nearest'
        self.ckpt = None  # for checkpointing models when using unet;
        self.save_iter = 15  # save model every this many iterations;
        self.recompute_CNN = False  # save memory using tf.recompute_grad;

        self.height_scale_factor = 4000  # to scale the ego_height parameter to avoid dealing with large values;

        # camera parameters for getting absolute scale;
        self.use_absolute_scale_calibration = False
        self.effective_focal_length_mm = 4.3  # effective focal length in mm;
        self.magnification_j = None  # user needs to figure out the magnification of the jth camera view;
        self.j = 0  # and specify which camera view index; by default, assume the first camera;

    def create_variables(self, deformation_model, learning_rates=None, variable_initial_values=None, recon=None,
                         normalize=None, remove_global_transform=False, antialiasing_filter=False,
                         stack_downsample_factor=None, force_ground_surface_up=False):
        # define tf.Variables and optimizers;
        # deformation_model: affects what tf.Variables will be defined and optimized;
        # learning_rates is a dictionary of variable names (strings) and the corresponding learning rate; None means
        # use the default, defined below; can specify a learning rate a negative number, indicating that variable is not
        # optimized; can supply initial variable values as a dictionary; if None, use default;
        # if recon and normalize are supplied, then initialize recon_previous and normalize_previous with these after
        # upsampling; both must be supplied; (this is for momentum/running average calculations); most likely, recon and
        # normalize will be from earlier reconstruction attempts using mesoSfM, but at lower resolution;
        # remove_global_transform: e.g., no global shift or scale;
        # stack_downsample_factor is an integer that downsamples the stack and coordinates; if None, then it will be
        # computed from self.scale;
        # force_ground_surface_up; when using camera model, force the mean surface normal to be [0,0,-1];

        # these must be both be None, or neither None:
        assert (self.batch_size is not None) == (self.momentum is not None)
        assert (recon is None) == (normalize is None)

        # define downsample factor:
        self.recon_shape = np.int32(self.recon_shape_base * self.scale)
        if stack_downsample_factor is None:
            self.downsample = np.int32(1 / self.scale)  # also downsample the images to save computation;
            self.downsample = np.maximum(self.downsample, 1)  # obviously can't downsample with 0;
        else:
            self.downsample = stack_downsample_factor
        self.im_downsampled_shape = np.array([(self.stack.shape[1] - 1) // self.downsample + 1,
                                              (self.stack.shape[2] - 1) // self.downsample + 1])

        # coordinates of the images:
        c = np.arange(self.stack.shape[2], dtype=np.uint16)
        r = np.arange(self.stack.shape[1], dtype=np.uint16)
        r, c = np.meshgrid(r, c, indexing='ij')
        rc_base = np.stack([r, c]).T
        self.rc_base = np.tile(rc_base[None],
                               [self.num_images, 1, 1, 1])  # base coordinates (could be downsampled after applying a
        # scale factor);

        self.deformation_model = deformation_model
        self.remove_global_transform = remove_global_transform
        self.force_ground_surface_up = force_ground_surface_up  # only relevant if using a camera model;

        # list of tf.Variables and optimizers, to be populated by one or more _create_variables_*deformation_model*;
        self.train_var_list = list()
        self.optimizer_list = list()
        self.non_train_list = list()  # list of variables that aren't trained (probably .assign()'d; for checkpoints);
        self.tensors_to_track = dict()  # intermediate tensors to track; have a tf.function return the contents;

        def use_default_for_missing(input_dict, default_dict):
            # to be used directly below; allows for dictionaries in which not all keys are specified; if not specified,
            # then use default_dict's value;

            if input_dict is None:  # if nothing given, then use the default;
                return default_dict
            else:
                for key in default_dict:
                    if key in input_dict:
                        if input_dict[key] is None:  # if the key is present, but None is specified;
                            input_dict[key] = default_dict[key]
                        else:  # i.e., use the value given;
                            pass
                    else:  # if key is not even present;
                        input_dict[key] = default_dict[key]
                return input_dict

        if 'camera_parameters' in deformation_model:
            if 'unet' in deformation_model:
                # user has to manually define these:
                assert self.filters_list is not None
                assert self.skip_list is not None

            default_learning_rates = {'camera_focal_length': 1e-3, 'camera_height': 1e-3, 'ground_surface_normal': 1e-3,
                                      'camera_in_plane_angle': 1e-3, 'rc': 10, 'gain': 1e-3, 'ego_height': 1e-3,
                                      'bias': 1e-3}
            default_variable_initial_values = {'camera_focal_length': np.float32(1),  # initialize same value as ...
                                               'camera_height': 1 * np.ones([self.num_images]),  # ...for height;
                                               'ground_surface_normal': np.concatenate(
                                                   [np.zeros((self.num_images, 2)) + 1e-7,  # to avoid /0;
                                                    -np.ones((self.num_images, 1))], axis=1),
                                               'camera_in_plane_angle': np.zeros([self.num_images]),
                                               'rc': self.ul_coords,
                                               'ego_height': 1e-7 +  # add small value to allow gradients to prop;
                                                         np.zeros([self.num_images,
                                                                   self.im_downsampled_shape[0],
                                                                   self.im_downsampled_shape[1]]),
                                               'gain': np.ones(self.num_images),
                                               'bias': np.zeros(self.num_images)}
            learning_rates = use_default_for_missing(learning_rates, default_learning_rates)
            variable_initial_values = use_default_for_missing(variable_initial_values, default_variable_initial_values)
            self._create_variables_camera_parameters(learning_rates, variable_initial_values)

            # these are for perspective to perspective and perspective to orthographic:
            if 'unet' not in deformation_model:
                # create radial deformation field:
                self._create_variables_perspective_to_orthographic(learning_rates, variable_initial_values)
            elif 'unet' in deformation_model:
                # create neural network layers:
                self._create_variables_height_map_unet(learning_rates, variable_initial_values)

            if self.remove_global_transform:
                # removing the global scaling transform entails adapting the focal length to the mean height;
                assert learning_rates['camera_focal_length'] < 0

            if 'perspective_to_perspective' in deformation_model:
                # these won't be optimized; also these initializing values are very temporary, and should be immediately
                # modified by code later; the reason why they are tf.Variables is so that the user can manually modify
                # them in eager mode with .assign();
                self.reference_camera_height = tf.Variable(1, dtype=self.tf_dtype, name='reference_camera_height')
                self.reference_camera_rc = tf.Variable(np.zeros(2), dtype=self.tf_dtype, name='reference_camera_rc')
                # note that because there's no tilt, reference_camera_height is the same as the camera to vanishing
                # point distance, and the reference_camera_rc is the same as the vanishing point position;
                self.non_train_list.append(self.reference_camera_rc)
                self.non_train_list.append(self.reference_camera_height)
        else:
            raise Exception('invalid deformation model: ' + deformation_model)

        # intensity adjustment (uniform within a given image for now):
        self.gain = tf.Variable(variable_initial_values['gain'], dtype=self.tf_dtype, name='gain')
        self.gain_optim = self.optimizer(learning_rate=learning_rates['gain'])
        self.train_var_list.append(self.gain)
        self.optimizer_list.append(self.gain_optim)

        # intensity bias (also uniform within each image):
        self.bias = tf.Variable(variable_initial_values['bias'], dtype=self.tf_dtype, name='bias')
        self.bias_optim = self.optimizer(learning_rate=learning_rates['bias'])
        self.train_var_list.append(self.bias)
        self.optimizer_list.append(self.bias_optim)

        # barrel or pincushion distortion correction:
        if 'radial_camera_distortion' in learning_rates:
            assert 'camera' in deformation_model  # this is explicitly a camera modeling option;
            if 'radial_camera_distortion' in variable_initial_values:
                camera_distortion_init = variable_initial_values['radial_camera_distortion']
                if np.ndim(camera_distortion_init) == 0:  # user is specifying there to be only one common parameter;
                    camera_distortion_init = np.reshape(camera_distortion_init, (1, 1))  # make sure at least 2 dims;
                elif np.ndim(camera_distortion_init) == 1:
                    # assume this one dimension refers to the camera view, not the polynomial order;
                    camera_distortion_init = np.reshape(camera_distortion_init, (-1, 1))
            else:
                camera_distortion_init = np.zeros((self.num_images, 1))  # second dim: polynomial order;
            num_poly_terms = camera_distortion_init.shape[1]  # number of terms in the polynomial (even only);
            self.radial_powers = (np.arange(num_poly_terms) + 1)  # half of the even powers to raise to;
            self.correct_radial_camera_distortion = camera_distortion_init.shape[0]  # this info used if batching;
            self.radial_camera_distortion = tf.Variable(camera_distortion_init, dtype=self.tf_dtype,
                                                        name='radial_camera_distortion')
            self.radial_camera_distortion_optim = self.optimizer(
                learning_rate=learning_rates['radial_camera_distortion'])
            self.train_var_list.append(self.radial_camera_distortion)
            self.optimizer_list.append(self.radial_camera_distortion_optim)
        else:
            self.correct_radial_camera_distortion = False

        if 'radial_camera_distortion_piecewise_linear' in learning_rates:
            assert 'camera' in deformation_model
            if 'radial_camera_distortion_piecewise_linear' in variable_initial_values:
                camera_distortion_init = variable_initial_values['radial_camera_distortion_piecewise_linear']
                assert np.ndim(camera_distortion_init) == 1  # for now, only allow a common distortion among all cams;
                # length of this vector determines how many discretization levels;
            else:
                camera_distortion_init = np.zeros(50)
            self.num_radial_pixels = len(camera_distortion_init)  # how many discretization levels (nodes);
            self.correct_radial_camera_distortion_piecewise_linear = -1  # used if batching images; this never equals
            # num_images;
            self.radial_camera_distortion_piecewise_linear = tf.Variable(camera_distortion_init, dtype=self.tf_dtype,
                                                                         name='radial_camera_distortion_piecewise_linear')
            self.radial_camera_distortion_piecewise_linear_optim = self.optimizer(
                learning_rate=learning_rates['radial_camera_distortion_piecewise_linear'])
            self.train_var_list.append(self.radial_camera_distortion_piecewise_linear)
            self.optimizer_list.append(self.radial_camera_distortion_piecewise_linear_optim)
        else:
            self.correct_radial_camera_distortion_piecewise_linear = False

        # if camera is not centered; this defines the center of the above two distortions, so center the camera first,
        # apply the distortions, then decenter back;
        if 'camera_distortion_center' in learning_rates:
            assert 'camera' in deformation_model
            if 'camera_distortion_center' in variable_initial_values:
                camera_distortion_center_init = variable_initial_values['camera_distortion_center']
                if np.ndim(camera_distortion_center_init) == 1:  # one common pair of parameters;
                    assert len(camera_distortion_center_init) == 2  # x and y centers;
                    camera_distortion_center_init = tf.reshape(camera_distortion_center_init, (1, -1))
                if np.ndim(camera_distortion_center_init) == 0:
                    raise Exception('must supply two values for x/y centration parameters')
            else:
                camera_distortion_center_init = np.zeros((self.num_images, 2))  # by default, diff pair for each camera;
            self.correct_camera_distortion_center = camera_distortion_center_init.shape[0]  # this info used if
            # batching;
            self.camera_distortion_center = tf.Variable(camera_distortion_center_init, dtype=self.tf_dtype,
                                                    name='camera_distortion_center')
            self.camera_distortion_center_optim = self.optimizer(
                learning_rate=learning_rates['camera_distortion_center'])
            self.train_var_list.append(self.camera_distortion_center)
            self.optimizer_list.append(self.camera_distortion_center_optim)
        else:
            self.correct_camera_distortion_center = False

        # create a list of booleans to accompany self.train_var_list and self.optimizer_list to specify whether to train
        # those variables (as specified by the whether the user-specified learning rates are negative); doing this so
        # that autograph doesn't traverse all branches of the conditionals; if the user ever wants to turn off
        # optimization of a variable mid-optimization, then just do .assign(0) to the learning rate, such that the
        # update is still happening, but the change is 0;
        self.trainable_or_not = list()
        for var in self.train_var_list:
            if type(var) is list:
                # if the variable is a list of variables, then this should be for the unet; modify here if there are
                # other scenarios;
                assert 'unet' in self.deformation_model
                name = 'ego_height'
            else:
                name = var.name[:-2]
            flag = learning_rates[name] > 0
            self.trainable_or_not.append(flag)

        # downsample rc coordinates and stack:
        rc = np.transpose(self.rc_base[:, ::self.downsample, ::self.downsample, :], (0, 2, 1, 3))
        if antialiasing_filter:
            downsample = int(self.downsample)  # cv2 doesn't like numpy values?
            if downsample == 1:
                print('warning: antialiasing filter is applied even though there is no downsampling')
            ksize = int(downsample * 2.5)
            if ksize % 2 == 0:  # must be odd;
                ksize += 1
            stack_downsamp = np.stack(
                [cv2.GaussianBlur(im, (ksize, ksize), downsample, downsample)
                 [::downsample, ::downsample] for im in self.stack])
        else:
            stack_downsamp = self.stack[:, ::self.downsample, ::self.downsample, :]

        rc_downsamp = np.reshape(rc, (self.rc_base.shape[0], -1, self.rc_base.shape[-1]))  # flatten spatial dims;
        stack_downsamp = np.reshape(stack_downsamp,
                                    [self.num_images, -1, self.num_channels])  # flatten spatial dims;
        self.rc_downsamp = rc_downsamp
        self.stack_downsamp = stack_downsamp

        # create variables relevant for batching (or set them to None if not batching):
        if self.momentum is not None:
            if 'camera_parameters_perspective_' in deformation_model:
                # we're going to give the coregistered height map a ride;
                num_channels = self.num_channels + 1
            else:
                num_channels = self.num_channels

            if recon is None:  # if none supplied, initialize with 0s;
                recon_previous = np.zeros([self.recon_shape[0], self.recon_shape[1], num_channels])
                normalize_previous = np.zeros(self.recon_shape)
            else:  # otherwise, upsample to the current shape;
                recon_previous = cv2.resize(np.nan_to_num(recon), tuple(self.recon_shape[::-1]))
                normalize_previous = cv2.resize(np.nan_to_num(normalize), tuple(self.recon_shape[::-1]))
                if num_channels == 1:
                    # cv2 seems to squeeze singleton channels dimensions, so if it's singleton, add it back:
                    recon_previous = recon_previous[:, :, None]
                if recon_previous.shape[-1] != num_channels:
                    # if you would like to use the previous RGB image as initialization, but your previous run didn't
                    # estimate a height map;
                    assert 'camera_parameters_perspective_' in deformation_model
                    # add empty height channel:
                    recon_previous = np.concatenate([recon_previous, np.zeros_like(recon_previous[:, :, 0:1])], axis=-1)

            # initialize first recon and normalize tensors for momentum; use the scaled recon shape, not the base shape;
            self.recon_previous = tf.Variable(recon_previous, dtype=self.tf_dtype, trainable=False)
            self.non_train_list.append(self.recon_previous)
        else:
            self.recon_previous = None
            self.normalize_previous = None

    def _create_variables_perspective_to_orthographic(self, learning_rates, variable_initial_values):
        # radially inwardly pointing vector magnitudes, where the larger the magnitude, the taller the object;

        ego_height = variable_initial_values['ego_height']
        # make sure the first dimensions match (number of images in stack):
        assert ego_height.shape[0] == self.num_images
        if ego_height.shape[1:] != tuple(self.im_downsampled_shape):
            # presumably you've initialized this with the results from another optimization at a different scale;
            # thus, resize to match the current scale:
            if type(ego_height) != np.ndarray:
                # convert from tf to np if you need to:
                ego_height = ego_height.numpy()
            ego_height = np.stack([cv2.resize(im, tuple(self.im_downsampled_shape[::-1])) for im in ego_height])

        self.ego_height = tf.Variable(ego_height, dtype=self.tf_dtype, name='ego_height')
        self.ego_height_optim = self.optimizer(learning_rate=learning_rates['ego_height'])
        self.train_var_list.append(self.ego_height)
        self.optimizer_list.append(self.ego_height_optim)

    def _create_variables_height_map_unet(self, learning_rates, variable_initial_values):

        self.network = unet(self.filters_list, self.skip_list, output_nonlinearity=self.output_nonlinearity,
                            upsample_method=self.upsample_method)
        # run the network once so that we can access network.trainable_variables
        self.network(tf.zeros([1, 2 ** len(self.filters_list), 2 ** len(self.filters_list), self.num_channels],
                              dtype=self.tf_dtype))

        self.train_var_list.append(self.network.trainable_variables)
        self.optimizer_list.append(self.optimizer(learning_rate=learning_rates['ego_height']))

        if self.recompute_CNN:
            self.network = tf.recompute_grad(self.network)

        # get padded shape that the network likes;
        self.padded_shape = [get_compatible_size(dim, len(self.filters_list)) for dim in self.im_downsampled_shape]
        pad_r = self.padded_shape[0] - self.im_downsampled_shape[0]
        pad_c = self.padded_shape[1] - self.im_downsampled_shape[1]
        pad_top = pad_r // 2
        pad_bottom = int(tf.math.ceil(pad_r / 2))
        pad_left = pad_c // 2
        pad_right = int(tf.math.ceil(pad_c / 2))
        pad_specs = ((pad_top, pad_bottom), (pad_left, pad_right))
        self.pad_layer = tf.keras.layers.ZeroPadding2D(pad_specs)
        self.depad_layer = tf.keras.layers.Cropping2D(pad_specs)

    def _create_variables_camera_parameters(self, learning_rates, variable_initial_values):
        # coordinates are in camera space;

        self.camera_focal_length = tf.Variable(variable_initial_values['camera_focal_length'], dtype=self.tf_dtype,
                                               name='camera_focal_length')
        self.camera_focal_length_optim = self.optimizer(learning_rate=learning_rates['camera_focal_length'])

        self.camera_height = tf.Variable(variable_initial_values['camera_height'], dtype=self.tf_dtype,
                                         name='camera_height')  # height from the camera perspective;
        self.camera_height_optim = self.optimizer(learning_rate=learning_rates['camera_height'])

        self.ground_surface_normal = tf.Variable(variable_initial_values['ground_surface_normal'], dtype=self.tf_dtype,
                                                 name='ground_surface_normal')
        self.ground_surface_normal_optim = self.optimizer(
            learning_rate=learning_rates['ground_surface_normal'])

        self.camera_in_plane_angle = tf.Variable(variable_initial_values['camera_in_plane_angle'],
                                                 dtype=self.tf_dtype, name='camera_in_plane_angle')
        self.camera_in_plane_angle_optim = self.optimizer(
            learning_rate=learning_rates['camera_in_plane_angle'])

        self.rc_ul_per_im = tf.Variable(variable_initial_values['rc'], dtype=self.tf_dtype, name='rc')
        self.rc_ul_per_im_optim = self.optimizer(learning_rate=learning_rates['rc'])

        self.train_var_list.append(self.camera_focal_length)
        self.optimizer_list.append(self.camera_focal_length_optim)
        self.train_var_list.append(self.camera_height)
        self.optimizer_list.append(self.camera_height_optim)
        self.train_var_list.append(self.ground_surface_normal)
        self.optimizer_list.append(self.ground_surface_normal_optim)
        self.train_var_list.append(self.camera_in_plane_angle)
        self.optimizer_list.append(self.camera_in_plane_angle_optim)
        self.train_var_list.append(self.rc_ul_per_im)
        self.optimizer_list.append(self.rc_ul_per_im_optim)

    def generate_dataset(self):
        # user calls this function to get a dataset to iterate over; if not using batching, then just return a tuple or
        # list of length 1 (i.e., the whole dataset is one batch);

        if self.batch_size is not None:
            if self.batch_across_images:
                # sample a subset of the images, and keep track of the indices downsampled so that you can gather the
                # corresponding variables;
                tensor_slices = (self.stack_downsamp,
                                 (self.rc_downsamp, np.arange(self.num_images, dtype=np.int32)))
                dataset = (tf.data.Dataset.from_tensor_slices(tensor_slices).shuffle(self.num_images)
                           .batch(self.batch_size, drop_remainder=True).repeat(None).prefetch(1))
                return dataset
            else:
                # transpose to batch along space, not image number;
                rc_downsamp_T = np.transpose(self.rc_downsamp, (1, 0, 2))
                stack_downsamp_T = np.transpose(self.stack_downsamp, (1, 0, 2))
                if 'camera_parameters_perspective_' in self.deformation_model:
                    # need to also get coordinates of the spatial positions to index into pixel-wise deformation fields:
                    tensor_slices = (stack_downsamp_T,
                                     (rc_downsamp_T, np.arange(np.prod(self.im_downsampled_shape), dtype=np.int32)))
                else:
                    tensor_slices = (stack_downsamp_T, rc_downsamp_T)
                dataset = (tf.data.Dataset.from_tensor_slices(tensor_slices)
                           .shuffle(len(rc_downsamp_T)).batch(self.batch_size).repeat(None).prefetch(1))
                return dataset
        else:  # basically a 1-batch dataset;
            return self.stack_downsamp, self.rc_downsamp

    def _warp_camera_parameters(self, rc_downsamp, use_radial_deformation, p2p_warp_mode=None,
                                inds_downsamp=None, stack_downsamp=None):
        # shape of rc_downsamp: num_images, _, 2;
        # use_radial_deformation is a boolean flag specifying whether to do the per-pixel radial deformation fields to
        # warp perspective to othographic OR perspective to perspective; if the latter, then p2p_warp_mode specifies
        # how to specify the perspective reference to warp to; the options are 'mean', 'random', 'fixed', and None,
        # where None means you're using perspective-to-orthographpic warping and not perspective to perspective;
        # inds_downsamp is passed if batching and using radial deformations;
        # stack_downsamp is only needed if using a unet;

        if p2p_warp_mode is None:
            assert 'perspective_to_orthographic' in self.deformation_model
        else:
            assert 'perspective_to_perspective' in self.deformation_model

        if self.batch_across_images and self.batch_size is not None:
            # in generate_recon, we defined the batch and non-batch versions;

            rc_ul_per_im = self.rc_ul_per_im_batch
            gain = self.gain_batch
            bias = self.bias_batch
            camera_height = self.camera_height_batch
            ground_surface_normal = self.ground_surface_normal_batch
            camera_in_plane_angle = self.camera_in_plane_angle_batch
            if 'unet' not in self.deformation_model:
                ego_height = self.ego_height_batch
                self.ego_height_to_regularize = self.ego_height_batch
            else:
                pass
            if self.correct_radial_camera_distortion:
                if self.correct_radial_camera_distortion == self.num_images:
                    radial_camera_distortion = self.radial_camera_distortion_batch
                else:
                    radial_camera_distortion = self.radial_camera_distortion
            if self.correct_radial_camera_distortion_piecewise_linear:
                if self.correct_radial_camera_distortion_piecewise_linear == self.num_images:
                    radial_camera_distortion_piecewise_linear = self.radial_camera_distortion_piecewise_linear_batch
                else:
                    radial_camera_distortion_piecewise_linear = self.radial_camera_distortion_piecewise_linear
            if self.correct_camera_distortion_center:
                if self.correct_camera_distortion_center == self.num_images:
                    camera_distortion_center = self.camera_distortion_center_batch
                else:
                    camera_distortion_center = self.camera_distortion_center
            num_images = self.batch_size  # for reshaping below;
            camera_focal_length = self.camera_focal_length  # not batched;
        else:
            rc_ul_per_im = self.rc_ul_per_im
            gain = self.gain
            bias = self.bias
            camera_height = self.camera_height
            ground_surface_normal = self.ground_surface_normal
            camera_in_plane_angle = self.camera_in_plane_angle
            if 'unet' not in self.deformation_model:
                ego_height = self.ego_height
                self.ego_height_to_regularize = self.ego_height
            if self.correct_radial_camera_distortion:
                radial_camera_distortion = self.radial_camera_distortion
            if self.correct_radial_camera_distortion_piecewise_linear:
                radial_camera_distortion_piecewise_linear = self.radial_camera_distortion_piecewise_linear
            if self.correct_camera_distortion_center:
                camera_distortion_center = self.camera_distortion_center

            num_images = self.num_images  # for reshaping below;
            camera_focal_length = self.camera_focal_length

        if self.remove_global_transform:
            # don't use self.camera_focal_length; set to geometric mean of the camera heights;
            # also, use the tf.Variable version always (not the batch version);
            camera_focal_length = tf.reduce_prod(self.camera_height, axis=0, keepdims=False) ** (1 / self.num_images)
            self.camera_focal_length.assign(camera_focal_length)
            camera_in_plane_angle = camera_in_plane_angle - tf.reduce_mean(self.camera_in_plane_angle)
            rc_ul_per_im = rc_ul_per_im - tf.reduce_mean(self.rc_ul_per_im, axis=0, keepdims=True)

        im_dims = np.array(self.stack.shape)[1:3]  # for normalization of image coordinates;
        max_dim = np.max(im_dims)  # to keep isotropic;
        camera_yx = (rc_downsamp - .5 * im_dims[None, None, :]) / max_dim  # normalize to -.5 to .5;

        if self.correct_camera_distortion_center:
            camera_yx -= camera_distortion_center[:, None, :]

        if self.correct_radial_camera_distortion:
            camera_r2 = camera_yx[:, :, 0] ** 2 + camera_yx[:, :, 1] ** 2  # radial distance squared;
            # dims^: camera, pixels
            camera_r2 *= 2  # make it go from -1 to 1 rather than -.5 to .5;

            if self.correct_radial_camera_distortion:
                # even polynomial to account for distortion:
                camera_even_poly = tf.math.pow(camera_r2[:, :, None], self.radial_powers[None, None, :])
                # dims^: camera, pixels, power
                camera_even_poly = tf.reduce_sum(camera_even_poly * radial_camera_distortion[:, None, :], 2)
                # dims^: camera, pixels
                radial_correction_factor = 1 + camera_even_poly[:, :, None]
                self.tensors_to_track['camera_distortion_radial'] = radial_correction_factor
            else:
                radial_correction_factor = 1
            camera_yx = camera_yx * radial_correction_factor

        if self.correct_radial_camera_distortion_piecewise_linear:
            camera_r = tf.sqrt(camera_yx[:, :, 0] ** 2 + camera_yx[:, :, 1] ** 2)  # radial distance;
            # dims^: camera, pixels; these radial coordinates should be between 0 and .5*sqrt(2), but could go higher
            # if th center moves); thus to be safe, just multiply by num_radial_pixels;
            r_scale = camera_r * self.num_radial_pixels
            # find nearest pixels and distances thereto:
            r_floor = tf.floor(r_scale)
            r_ceil = tf.minimum(r_floor + 1, self.num_radial_pixels - 1)  # to prevent out of range indexing;
            r_middle = r_scale - r_floor
            r_floor = tf.cast(r_floor, dtype=tf.int32)
            r_ceil = tf.cast(r_ceil, dtype=tf.int32)
            distortion = 1 + radial_camera_distortion_piecewise_linear
            distortion /= tf.reduce_max(distortion)  # to prevent global expansion;
            correction_factor_floor = tf.gather(distortion, r_floor)
            correction_factor_ceil = tf.gather(distortion, r_ceil)
            # bilinear interpolation:
            correction_factor = correction_factor_ceil * r_middle + correction_factor_floor * (1 - r_middle)
            camera_yx *= correction_factor[:, :, None]
            self.tensors_to_track['camera_distortion_piecewise_linear'] = correction_factor

        if self.correct_camera_distortion_center:
            camera_yx += camera_distortion_center[:, None, :]

        # in-plane rotation:
        cos = tf.cos(camera_in_plane_angle)
        sin = tf.sin(camera_in_plane_angle)
        rotmat_xy = tf.stack([[cos, sin], [-sin, cos]])
        camera_yx = tf.einsum('cri,ijc->crj', camera_yx, rotmat_xy)

        n_ground, _ = tf.linalg.normalize(ground_surface_normal, axis=1)  # normalize to unit mag;
        # shape^: num_images, 3

        # projecting to object space (computed analytically and taylor-expanded);
        nx = n_ground[:, 0][:, None, None]
        ny = n_ground[:, 1][:, None, None]
        nx2 = nx ** 2
        ny2 = ny ** 2
        x = camera_yx[:, :, 1][:, :, None]
        y = camera_yx[:, :, 0][:, :, None]
        h = camera_height[:, None, None]
        f = camera_focal_length
        # using a taylor expansion:
        flat_x = (h * x / f +
                  h * x * (nx * x + ny * y) / f ** 2 +
                  h * (f ** 2 * nx2 * x + 2 * nx2 * x ** 3 + f ** 2 * nx * ny * y + 4 * nx * ny * x ** 2 * y +
                       2 * ny2 * x * y ** 2) / 2 / f ** 3)
        flat_y = (h * y / f +
                  h * y * (nx * x + ny * y) / f ** 2 +
                  h * (f ** 2 * ny2 * y + 2 * ny2 * y ** 3 + f ** 2 * nx * ny * x + 4 * nx * ny * y ** 2 * x +
                       2 * nx2 * y * x ** 2) / 2 / f ** 3)
        flat_xy = tf.concat([flat_x, flat_y], axis=2)

        n_dot_r = n_ground[:, 2] * camera_height  # shape: num_images; dot product between a point on the
        # ground, r (0, 0, camera_height); (this is needed below);
        self.flat_xy = flat_xy

        if use_radial_deformation:
            # compute the vanishing point, which will be used if you use the projective to orthographic mode;
            vanish_xyz = n_dot_r[:, None] * n_ground  # shape: num_images, 3;
            camera_to_vanish_point_xyz = tf.norm(vanish_xyz, axis=1)  # distance from camera to the ground; the actual
            # height of the camera;

            # projection to object space simplifies to this:
            vanish_xy = -camera_height[:, None] * n_ground[:, :2]

            # vanishing point in camera plane:
            vanish_camera_xyz = n_ground * camera_focal_length / n_ground[:, 2:]  # follow surface normal to camera
            # plane;
            vanish_camera_xy = vanish_camera_xyz[:, 0:2]  # don't need z, it's just the focal length;
            # account for in-plane camera rotation:
            vanish_camera_xy = tf.einsum('ci,ijc->cj', vanish_camera_xy, rotmat_xy)
            self.vanish_xy = vanish_xy

        # convert back to row-column:
        flat_rc = (flat_xy[:, :, ::-1] * max_dim + .5 * im_dims[None, None, :])
        if use_radial_deformation:
            vanish_rc = (vanish_xy[:, ::-1] * max_dim + .5 * im_dims[None, :])
            vanish_camera_rc = (vanish_camera_xy[:, ::-1] * max_dim + .5 * im_dims[None, :])
            self.camera_to_vanish_point_rc = camera_to_vanish_point_xyz * max_dim  # convert from xy units to rc units;
            self.tensors_to_track['camera_to_vanish_point_xyz'] = self.camera_to_vanish_point_rc

        # add translations (same as for the homography and affine implementations):
        # these translations don't affect camera_to_vanish_point;
        rc_warp = rc_ul_per_im[:, None, :] + flat_rc
        if use_radial_deformation:
            vanish_warp = rc_ul_per_im + vanish_rc
        if self.restrict_function == 'mod_with_random_shifts':
            # to discourage registration with overlapped regions;
            random_shift = tf.random.uniform(shape=(1, 1, 2), minval=0, maxval=self.recon_shape.max() / self.scale)
            rc_warp += random_shift
            if use_radial_deformation:
                vanish_warp += random_shift[0]
        else:
            rc_warp += self.ul_offset[None, None, :]
            if use_radial_deformation:
                vanish_warp += self.ul_offset[None, :]

        if 'unet' in self.deformation_model:
            # generate self.ego_height; doesn't matter if batching or not because it's generated from the image batch;
            unet_input = tf.reshape(stack_downsamp, [num_images, self.im_downsampled_shape[0],
                                                     self.im_downsampled_shape[1], self.num_channels])

            unet_input = self.pad_layer(unet_input)
            unet_output = self.network(unet_input)
            ego_height = tf.reduce_mean(self.depad_layer(unet_output), [-1])  # remove last dimension;
            ego_height *= self.unet_scale
            self.ego_height_to_regularize = ego_height  # need this for regularization;
            self.ego_height = ego_height

        # multiplicative version:
        if use_radial_deformation:
            self.tensors_to_track['vanish_warp'] = vanish_warp
            self.tensors_to_track['vanish_camera'] = vanish_camera_rc
            if p2p_warp_mode is None:  # using perspective-to-orthographic
                ego_height_flat = tf.reshape(ego_height, [num_images, -1])  # flatten spatial dimensions;
                if self.subtract_min_from_height_map:
                    ego_height_flat -= tf.reduce_min(ego_height_flat)
                if inds_downsamp is not None:
                    assert self.batch_size is not None  # this should never be raised, but just in case;
                    ego_height = tf.gather(ego_height_flat, inds_downsamp, axis=1)  # batch along pixels;
                else:
                    ego_height = ego_height_flat

                if self.use_absolute_scale_calibration:

                    H = self.camera_to_vanish_point_rc[:, None]

                    if self.batch_size is not None:
                        # need to compute the self.j'th camera-to-vanish-point height in case you're batching, which means
                        # that the self.j'th entry may not be computed;
                        n_ground, _ = tf.linalg.normalize(self.ground_surface_normal[self.j], axis=0)
                        n_dot_r = n_ground[2] * self.camera_height[self.j]
                        vanish_xyz = n_dot_r * n_ground  # shape: 3;
                        camera_to_vanish_point_xyz = tf.norm(vanish_xyz, axis=0)
                        H_j = camera_to_vanish_point_xyz * max_dim
                    else:
                        H_j = H[self.j]

                    r = rc_warp - vanish_warp[:, None, :]  # lateral distance to vanishing point;
                    M_j = self.magnification_j
                    f_eff = self.effective_focal_length_mm
                    self.another_height_scale_factor = f_eff * (1 + 1 / M_j)  # scale ego_height again to make this
                    # case similar to the other case (self.use_absolute_scale_calibration);
                    h = ego_height * self.another_height_scale_factor

                    delta_radial = h / f_eff / (1 + 1 / M_j * H / H_j)
                    rc_warp = r * (1 - delta_radial[:, :, None]) + vanish_warp[:, None, :]

                    ego_height *= self.height_scale_factor  # to keep consistent with regularization coefficients;
                    # note that you have to divide by height_scale_factor because the height map is scaled by this, but
                    # you have to divide by another_scale_factor because the multiplication above allows ego_height to
                    # shrink;
                else:
                    ego_height *= self.height_scale_factor  # denominator is large in next line, so multiply by a large
                    # value to allow self.ego_height to take on smaller values;

                    delta_radial = ego_height / self.camera_to_vanish_point_rc[:, None]
                    rc_warp = ((rc_warp - vanish_warp[:, None, :]) * (1 - delta_radial[:, :, None]) +
                               vanish_warp[:, None, :])
                self.ego_height_for_concat = ego_height  # for concatenating with self.im below;
            else:  # perspective-to-perspective warping
                # first, need to define the reference camera view:
                if p2p_warp_mode == 'mean':
                    self.reference_camera_height.assign(tf.reduce_mean(self.camera_to_vanish_point_rc))
                    self.reference_camera_rc.assign(tf.reduce_mean(vanish_warp, axis=0))
                elif p2p_warp_mode == 'random':
                    height_min = tf.reduce_min(self.camera_to_vanish_point_rc)
                    height_max = tf.reduce_max(self.camera_to_vanish_point_rc)
                    self.reference_camera_height.assign(tf.random.uniform((), height_min, height_max))
                    rc_min = tf.reduce_min(vanish_warp, axis=0)
                    rc_max = tf.reduce_max(vanish_warp, axis=0)
                    self.reference_camera_rc.assign(tf.random.uniform((2,), rc_min, rc_max))
                elif p2p_warp_mode == 'fixed':
                    pass  # do nothing, accept current values;
                elif p2p_warp_mode == 'random_choice':
                    # pick one of the camera view among the existing;
                    random_choice = tf.random.uniform((), 0, self.num_images, dtype=tf.int32)
                    self.reference_camera_height.assign(tf.gather(self.camera_to_vanish_point_rc, random_choice))
                    self.reference_camera_rc.assign(tf.gather(vanish_warp, random_choice))
                else:
                    raise Exception('invalid perspective-to-perspective warp mode passed to gradient_update: '
                                    + p2p_warp_mode)
                # vector deformation field to warp to the reference perspective:
                h = tf.reshape(ego_height, [num_images, -1, 1]) * self.height_scale_factor
                H_r = self.reference_camera_height
                H = self.camera_to_vanish_point_rc[:, None, None]
                R_r = self.reference_camera_rc[None, None, :]
                R = vanish_warp[:, None, :]
                r = rc_warp - R  # position vectors relative to each camera's vanishing point;
                p2p_warp = h / (H_r - h) * (R - R_r) + r * h * (H - H_r) / H / (H_r - h)  # the magic equation;
                rc_warp += p2p_warp  # shape: num_images, flattened spatial, 2

                self.ego_height_for_concat = h  # for concatenating with self.im below;

                if self.use_absolute_scale_calibration:
                    raise Exception('not yet implemented for perspective-to-perspective')

        rc_warp = tf.reshape(rc_warp, [-1, 2]) * self.scale  # flatten

        return rc_warp

    def _generate_recon(self, stack_downsamp, rc_downsamp, dither_coords, p2p_warp_mode=None, assign_update_recon=True):
        # backprojects all the images into the reconstruction, with the specified scale;
        # if batching, the gradient_update function will update the reconstruction with a running average;
        # if batching, this function should not be called by the user, as it will continually update the recon with the
        # same batch; if not batching, then this generates the full reconstruction;
        # p2p_warp_mode: if using perspective-to-perspective warping; can be None, 'mean', 'random', or 'fixed';
        # assign_update_recon: only relevant if using batching; controls whether to use the .assign() mechanism to
        # update the reconstruction (specified via update_gradient option in the gradient_update function);

        if self.batch_size is not None:
            if self.batch_across_images:
                # distinguish inds_downsamp and inds_image_downsamp, where the former is for pixel-level batching while
                # the latter is for image-level batching;
                rc_downsamp, inds_image_downsamp = rc_downsamp  # unpack

                # now, for all variables whose first dimension corresponds to the image dimension, gather:
                self.rc_ul_per_im_batch = tf.gather(self.rc_ul_per_im, inds_image_downsamp, axis=0)
                self.gain_batch = tf.gather(self.gain, inds_image_downsamp, axis=0)
                self.bias_batch = tf.gather(self.bias, inds_image_downsamp, axis=0)
                # these are used below:
                gain = self.gain_batch
                bias = self.bias_batch

                if 'camera_parameters' in self.deformation_model:
                    self.camera_height_batch = tf.gather(self.camera_height, inds_image_downsamp, axis=0)
                    self.ground_surface_normal_batch = tf.gather(self.ground_surface_normal,
                                                                 inds_image_downsamp, axis=0)
                    self.camera_in_plane_angle_batch = tf.gather(self.camera_in_plane_angle,
                                                                 inds_image_downsamp, axis=0)
                    if 'unet' not in self.deformation_model:
                        # if using unet, then ego_height will already be gathered as it is generated by the unet;
                        self.ego_height_batch = tf.gather(self.ego_height, inds_image_downsamp, axis=0)
                    # the following self.correct__ variables serve two purposes: 1) to signify whether they are being
                    # used, and 2) specify length of first dimension of the corresponding distortion variable to decide
                    # whether we need to use tf.gather;
                    if self.correct_radial_camera_distortion == self.num_images:
                        self.radial_camera_distortion_batch = tf.gather(self.radial_camera_distortion,
                                                                  inds_image_downsamp, axis=0)
                    if self.correct_radial_camera_distortion_piecewise_linear == self.num_images:
                        self.radial_camera_distortion_piecewise_linear_batch = tf.gather(
                            self.radial_camera_distortion_piecewise_linear, inds_image_downsamp, axis=0)
                    if self.correct_camera_distortion_center == self.num_images:
                        self.camera_distortion_center_batch = tf.gather(self.camera_distortion_center,
                                                                        inds_image_downsamp, axis=0)
                else:
                    raise Exception('image-level batching not yet implemented for a non-camera model')

                inds_downsamp = None
            else:
                # if batching, then stack_downsamp and rc_downsamp are transposed and need to be untransposed;

                if 'camera_parameters' in self.deformation_model:
                    # also need the indices of the pixels chosen, because radial deformations are pixel-wise;
                    # make sure you package these together into a tuple in the script;
                    rc_downsamp, inds_downsamp = rc_downsamp
                else:
                    inds_downsamp = None

                stack_downsamp = tf.transpose(stack_downsamp, (1, 0, 2))
                rc_downsamp = tf.transpose(rc_downsamp, (1, 0, 2))

                # these are used below:
                gain = self.gain
                bias = self.bias
        else:
            inds_downsamp = None
            # these are used below:
            gain = self.gain
            bias = self.bias
        # to save CPU memory, the dataset and coordinates are stored as uint8 and uint16, respectively; thus, cast to
        # float here;
        stack_downsamp = tf.cast(stack_downsamp, self.tf_dtype)
        rc_downsamp = tf.cast(rc_downsamp, self.tf_dtype)

        # function that restricts coordinates to the grid (store as self.variable so that error_map can use it):
        if self.restrict_function == 'clip':
            self.restrict = lambda x: tf.clip_by_value(x, tf.zeros_like(x), self.recon_shape[None] - 1)
        elif 'mod' in self.restrict_function:  # 'mod' or 'mod_with_random_shifts';
            self.restrict = lambda x: tf.math.floormod(x, self.recon_shape[None])
        else:
            raise Exception('invalid restrict_function')

        # apply gain:
        gain_norm = gain / tf.reduce_mean(gain)  # normalize so that there's no global gain;
        im = stack_downsamp * gain_norm[:, None, None] + bias[:, None, None]

        self.im = tf.reshape(im, (-1, self.num_channels))  # flatten all but channels;

        # warped coordinates:
        if self.deformation_model == 'camera_parameters':
            self.rc_warp = self._warp_camera_parameters(rc_downsamp, use_radial_deformation=False)
        elif self.deformation_model == 'camera_parameters_perspective_to_orthographic':
            self.rc_warp = self._warp_camera_parameters(rc_downsamp, inds_downsamp=inds_downsamp,
                                                        use_radial_deformation=True)
        elif self.deformation_model == 'camera_parameters_perspective_to_orthographic_unet':
            self.rc_warp = self._warp_camera_parameters(rc_downsamp, inds_downsamp=inds_downsamp,
                                                        use_radial_deformation=True, stack_downsamp=stack_downsamp)
        elif self.deformation_model == 'camera_parameters_perspective_to_perspective':
            self.rc_warp = self._warp_camera_parameters(rc_downsamp, p2p_warp_mode=p2p_warp_mode,
                                                        inds_downsamp=inds_downsamp,
                                                        use_radial_deformation=True)
        elif self.deformation_model == 'camera_parameters_perspective_to_perspective_unet':
            self.rc_warp = self._warp_camera_parameters(rc_downsamp, p2p_warp_mode=p2p_warp_mode,
                                                        inds_downsamp=inds_downsamp,
                                                        use_radial_deformation=True, stack_downsamp=stack_downsamp)
        else:
            raise Exception('invalid deformation model: ' + self.deformation_model)

        if 'camera_parameters_perspective_to_' in self.deformation_model:
            # adding the height map as a channel to the reconstruction, so first augment self.im with self.ego_height:
            self.im = tf.concat([self.im, tf.reshape(self.ego_height_for_concat, [-1])[:, None]], axis=1)

            self.num_channels_recon = self.num_channels + 1  # for the recon, need one more channel;
        else:
            self.num_channels_recon = self.num_channels

        #
        if dither_coords:
            self.rc_warp += tf.random.uniform([1, 2], -1, 1, dtype=self.tf_dtype)
            if self.batch_size is not None:
                print('Minor warning: using a running average for the recon while dithering coordinates')

        # neighboring pixels:
        rc_floor = tf.floor(self.rc_warp)
        rc_ceil = rc_floor + 1

        # distance to neighboring pixels:
        frc = self.rc_warp - rc_floor
        crc = rc_ceil - self.rc_warp

        # cast
        rc_floor = tf.cast(rc_floor, tf.int32)
        rc_ceil = tf.cast(rc_ceil, tf.int32)

        self.rc_ff = self.restrict(rc_floor)
        self.rc_cc = self.restrict(rc_ceil)
        self.rc_cf = self.restrict(tf.stack([rc_ceil[:, 0], rc_floor[:, 1]], 1))
        self.rc_fc = self.restrict(tf.stack([rc_floor[:, 0], rc_ceil[:, 1]], 1))

        # sig_proj = .42465  # chosen so that if point is exactly in between
        # ...two pixels, .5 weight is assigned to each pixel
        self.frc = tf.exp(-frc ** 2 / 2. / self.sig_proj ** 2)
        self.crc = tf.exp(-crc ** 2 / 2. / self.sig_proj ** 2)

        # augmented coordinates:
        rc_4 = tf.concat([self.rc_ff, self.rc_cc, self.rc_cf, self.rc_fc], 0)

        # interpolated:
        im_4 = tf.concat([self.im * self.frc[:, 0, None] * self.frc[:, 1, None],
                          self.im * self.crc[:, 0, None] * self.crc[:, 1, None],
                          self.im * self.crc[:, 0, None] * self.frc[:, 1, None],
                          self.im * self.frc[:, 0, None] * self.crc[:, 1, None]], 0)
        w_4 = tf.concat([self.frc[:, 0] * self.frc[:, 1],
                         self.crc[:, 0] * self.crc[:, 1],
                         self.crc[:, 0] * self.frc[:, 1],
                         self.frc[:, 0] * self.crc[:, 1]], 0)

        if self.momentum is not None:
            # update with moving average:
            self.im_4_previous = tf.gather_nd(self.recon_previous,
                                              rc_4) * w_4[:, None]  # with appropriate weighting by w_4;
            self.im_4_updated = (im_4 * self.momentum + self.im_4_previous * (1 - self.momentum))
            normalize = tf.scatter_nd(rc_4, w_4, self.recon_shape)
            self.norm_updated_regathered = tf.gather_nd(normalize, rc_4)
            self.im_4_updated_norm = self.im_4_updated / self.norm_updated_regathered[:, None]  # pre-normalize;
            # since tensor_scatter_nd_update doesn't accumulate values, but tensor_scatter_nd_add does, first zero
            # out the regions to be updated and then just add them:
            recon_zeroed = tf.tensor_scatter_nd_update(self.recon_previous, rc_4,
                                                       tf.zeros_like(self.im_4_updated_norm))
            self.recon = tf.tensor_scatter_nd_add(recon_zeroed, rc_4, self.im_4_updated_norm)
            if assign_update_recon:
                with tf.device('/CPU:0'):
                    self.recon_previous.assign(self.recon)

            self.normalize = None  # normalize not needed; in fact, normalize_previous also not needed;

        else:
            self.normalize = tf.scatter_nd(rc_4, w_4, self.recon_shape)
            self.recon = tf.scatter_nd(rc_4, im_4, [self.recon_shape[0], self.recon_shape[1], self.num_channels_recon])
            self.recon = tf.math.divide_no_nan(self.recon, self.normalize[:, :, None])  # creates recon H by W by C;

        if 'camera_parameters_perspective_to' in self.deformation_model:
            self.height_map = self.recon[:, :, -1]
            if self.use_absolute_scale_calibration:
                # divide out the scale factors to get the true height in mm:
                self.tensors_to_track['height_map'] = self.height_map / (self.height_scale_factor /
                                                                         self.another_height_scale_factor)
            else:
                self.tensors_to_track['height_map'] = self.height_map

    def _forward_prediction(self):
        # given the reconstruction, generate forward prediction;

        # forward model:
        ff = tf.gather_nd(self.recon, self.rc_ff)
        cc = tf.gather_nd(self.recon, self.rc_cc)
        cf = tf.gather_nd(self.recon, self.rc_cf)
        fc = tf.gather_nd(self.recon, self.rc_fc)

        self.forward = (ff * self.frc[:, 0, None] * self.frc[:, 1, None] +
                        cc * self.crc[:, 0, None] * self.crc[:, 1, None] +
                        cf * self.crc[:, 0, None] * self.frc[:, 1, None] +
                        fc * self.frc[:, 0, None] * self.crc[:, 1, None])

        self.forward /= ((self.frc[:, 0, None] * self.frc[:, 1, None]) +
                         (self.crc[:, 0, None] * self.crc[:, 1, None]) +
                         (self.crc[:, 0, None] * self.frc[:, 1, None]) +
                         (self.frc[:, 0, None] * self.crc[:, 1, None]))

        if 'camera_parameters_perspective' in self.deformation_model:
            # split off the last dimension, the height dimension, to compute the height map MSE:
            self.forward_height = self.forward[:, -1]
            self.error_height = self.forward_height - self.im[:, -1]  # save this for computing error map;
            self.MSE_height = tf.reduce_mean(self.error_height ** 2)

            self.error = self.forward[:, :-1] - self.im[:, :-1]  # remaining channels are the actual recon;
            self.MSE = tf.reduce_mean(self.error ** 2)
            self.recon = self.recon[:, :, :-1]  # discard the height map channel, as it's recorded elsewhere;
        else:
            self.error = self.forward - self.im  # save this for computing error map;
            self.MSE = tf.reduce_mean(self.error ** 2)

        if self.report_error_map:
            # project the squared error onto the reconstruction space;
            assert self.momentum is None  # for now, don't use this with batching/momentum;
            # don't have to use interpolated projection, because we don't need to compute gradients for this;

            with tf.device('/CPU:0'):
                rc_warp_int = self.restrict(tf.cast(self.rc_warp, tf.int32))
                self.error_map = tf.scatter_nd(rc_warp_int, self.error ** 2,
                                               [self.recon_shape[0], self.recon_shape[1], self.num_channels])
                norm = tf.scatter_nd(rc_warp_int, tf.ones(self.rc_warp.shape[0], dtype=self.tf_dtype), self.recon_shape)
                self.error_map = tf.math.divide_no_nan(self.error_map, norm[:, :, None])
        else:
            self.error_map = None

    def _add_regularization_loss(self, reg_coefs):

        # tf Variable to be regularized:
        if 'perspective_to_' in self.deformation_model:
            field = self.ego_height_to_regularize

        self.loss_list = [self.MSE]
        if 'L2' in reg_coefs and reg_coefs['L2'] is not None:
            self.loss_list.append(reg_coefs['L2'] * tf.reduce_sum(field ** 2) / self.scale ** 2)
        if 'L1' in reg_coefs and reg_coefs['L1'] is not None:
            self.loss_list.append(reg_coefs['L1'] * tf.reduce_sum(tf.sqrt(field ** 2 + 1e-7)) / self.scale ** 2)
        if 'TV2' in reg_coefs and reg_coefs['TV2'] is not None:  # TV squared;
            d0 = field[:, 1:, :-1] - field[:, :-1, :-1]
            d1 = field[:, :-1, 1:] - field[:, :-1, :-1]
            self.loss_list.append(reg_coefs['TV2'] * tf.reduce_sum(d0 ** 2 + d1 ** 2) / self.scale ** 2)
        if 'TV' in reg_coefs and reg_coefs['TV'] is not None:
            d0 = field[:, 1:, :-1] - field[:, :-1, :-1]
            d1 = field[:, :-1, 1:] - field[:, :-1, :-1]
            self.loss_list.append(reg_coefs['TV'] * tf.reduce_sum(tf.sqrt(d0 ** 2 + d1 ** 2 + 1e-7)) / self.scale ** 2)
        if 'height_map' in reg_coefs and reg_coefs['height_map'] is not None:
            # regularize consistency of deformations to the global height map;
            self.loss_list.append(reg_coefs['height_map'] * self.MSE_height)

    @tf.function
    def gradient_update(self, stack_downsamp, rc_downsamp, update_gradient=True, reg_coefs=None, dither_coords=True,
                        return_tracked_tensors=False, stop_gradient=True, p2p_warp_mode=None, return_grads=False,
                        return_loss_only=False):
        # do one gradient update step; pass thru the downsampled stack and rc coordinates via tf.datasets if batching;
        # one might set update_gradient to False if using batching and just want to create the reconstruction without
        # registering simultaneously;
        # dither_coords: if True, add a random float between -1 and 1 to prevent adaptation to the interpolation scheme;
        # stop_gradient: whether to use tf.stop_gradient on the reconstruction before computing the forward prediction;
        # p2p_warp_mode: see warp_camera_parameters and/or generate_recon; if using perspective-to-orthographic warping,
        # then set this to None; if using perspective-to-perspective warping, then set to one of 'mean', 'random',
        # 'fixed';
        # return_loss_only: as the name suggests;

        with tf.GradientTape() as tape:
            self._generate_recon(stack_downsamp, rc_downsamp, dither_coords, p2p_warp_mode, update_gradient)
            if stop_gradient:
                self.recon = tf.stop_gradient(self.recon)
            self._forward_prediction()

            if reg_coefs is not None:
                self._add_regularization_loss(reg_coefs)
                loss = tf.reduce_sum(self.loss_list)
            else:
                loss = self.MSE

        grads = tape.gradient(loss, self.train_var_list)
        # apply gradient update for each optimizer:
        if update_gradient:
            for grad, var, optimizer, train in zip(grads, self.train_var_list, self.optimizer_list,
                                                   self.trainable_or_not):
                if train:  # if user specifies negative learning rate, then don't update;
                    if type(var) is list or type(var) is ListWrapper:  # sometimes it's a ListWrapper; after restoring
                        # from checkpoint?
                        # this is probably the neural network variable list;
                        optimizer.apply_gradients(zip(grad, var))
                    else:
                        optimizer.apply_gradients([(grad, var)])

            # update other stuff:
            if 'camera_parameters' in self.deformation_model and self.force_ground_surface_up and update_gradient:
                # rotate all vectors such that their mean direction points in [0,0,-1];
                # (mean across all, not just batch!);
                # using rodrigues formula;
                n_mean = tf.reduce_mean(self.ground_surface_normal, axis=0)
                rot_axis = tf.stack([-n_mean[1], n_mean[0], 0])  # cross product;
                rot_axis, _ = tf.linalg.normalize(rot_axis)
                cos_rot_angle = -n_mean[2] / tf.norm(n_mean)
                sin_rot_angle = tf.sqrt(1 - cos_rot_angle ** 2 + 1e-6)  # prevent numerical issues;
                K = tf.stack([[0, -rot_axis[2], rot_axis[1]],
                              [rot_axis[2], 0, -rot_axis[0]],
                              [-rot_axis[1], rot_axis[0], 0]])
                rotmat = tf.eye(3) + sin_rot_angle * K + K @ K * (1 - cos_rot_angle)
                self.ground_surface_normal.assign(tf.einsum('ij,cj->ci', rotmat, self.ground_surface_normal))

        if return_loss_only:
            if reg_coefs is not None:
                return self.loss_list
            else:
                return self.MSE
        else:
            if reg_coefs is not None:
                return_list = [self.loss_list, self.recon, self.normalize, self.error_map]
            else:
                return_list = [self.MSE, self.recon, self.normalize, self.error_map]

            if return_tracked_tensors:
                return_list.append(self.tensors_to_track)
            if return_grads:
                return_list.append(grads)

            return return_list

    def checkpoint_all_variables(self, path='./tf_ckpts', skip_saving=False):
        # save checkpoints to restore in case unet diverges;
        # skip_saving if you just want to create the ckpt and manager;
        if self.ckpt is None:
            self.ckpt = tf.train.Checkpoint()
            self.ckpt.var = self.train_var_list
            self.ckpt.opt = self.optimizer_list
            self.ckpt.non_train_var = self.non_train_list
            self.manager = tf.train.CheckpointManager(self.ckpt, path, max_to_keep=2)
            # only keep two, restore the oldest;
        if not skip_saving:
            self.manager.save()

    def restore_all_variables(self):
        self.ckpt.restore(self.manager.checkpoints[0])

    @tf.function
    def get_all_variables(self):
        # return as dictionary so it can be used as initialization:
        with tf.device('/CPU:0'):
            # don't return neural network parameters -- use checkpoints instead;
            vars = dict()
            for var in self.train_var_list:
                try:
                    vars[var.name[:-2]] = var
                except AttributeError:  # ListWrapper for unet, for which we don't need the variables;
                    pass
            return vars


# u-net for mapping images to height map:
class unet(tf.keras.Model):
    def __init__(self, filters_list, skip_list, output_nonlinearity, upsample_method='bilinear'):
        # filters_list and skip_list are lists of number of filters in the upsample/downsample layers,
        # and the number of filters in the skip connections;
        # output_nonlinearity can be 'leaky_relu' or 'linear';
        # upsample_method can be 'bilinear' or 'nearest', which is used in the upsample blocks;
        super(unet, self).__init__()
        assert len(filters_list) == len(skip_list)
        self.filters_list = filters_list
        self.skip_list = skip_list
        self.output_nonlinearity = output_nonlinearity
        self.upsample_method = upsample_method
        self._build()

    def _build(self):
        # define all the layers of the unet;
        self.downsample_list = list()  # stores list of downsample blocks;
        self.downsample_skip_block_list = list()  # stores list of skip convolutional blocks;
        self.upsample_list = list()  # stores list of upsample blocks;
        self.upsample_concat_list = list()  # stores list of concatenation layers;

        # downsampling half:
        for num_filters, num_skip_filters in zip(self.filters_list, self.skip_list):
            self.downsample_list.append(self._downsample_block(num_filters))  # add to list of layers
            self.downsample_skip_block_list.append(self._skip_block(num_skip_filters))

        # upsampling half:
        for i, (num_filters, num_skip_filters) in enumerate(zip(self.filters_list[::-1], self.skip_list[::-1])):
            if num_skip_filters != 0:
                self.upsample_concat_list.append(tf.keras.layers.Concatenate())
            else:
                self.upsample_concat_list.append(None)  # as a placeholder
            if i == len(self.filters_list) - 1:
                # last block, use the specified output nonlinearity:
                self.upsample_list.append(self._upsample_block(num_filters,
                                                               nonlinearity=self.output_nonlinearity))
            else:
                self.upsample_list.append(self._upsample_block(num_filters))

    def _downsample_block(self, numfilters, kernel_size=3):
        return [tf.keras.layers.Conv2D(filters=numfilters, kernel_size=kernel_size,
                                       strides=(2, 2), padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.LeakyReLU(),
                # repeat, but no downsample this time
                tf.keras.layers.Conv2D(filters=numfilters, kernel_size=3,
                                       strides=(1, 1), padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.LeakyReLU()]

    def _upsample_block(self, numfilters, kernel_size=3, nonlinearity='leaky_relu'):
        layers_list = [tf.keras.layers.UpSampling2D(interpolation=self.upsample_method),
                       tf.keras.layers.Conv2D(filters=numfilters, kernel_size=kernel_size,
                                              strides=(1, 1), padding='same'),
                       tf.keras.layers.BatchNormalization(),
                       tf.keras.layers.LeakyReLU(),
                       tf.keras.layers.Conv2D(filters=numfilters, kernel_size=1,
                                              strides=(1, 1), padding='same'),  # kernel size 1
                       tf.keras.layers.BatchNormalization()]
        if nonlinearity == 'leaky_relu':
            layers_list.append(tf.keras.layers.LeakyReLU())
        elif nonlinearity == 'linear':
            pass
        else:
            raise Exception('invalid nonlinearity')
        return layers_list

    def _skip_block(self, numfilters=4, kernel_size=1):
        if numfilters == 0:  # no skip connections
            return None
        else:
            return [tf.keras.layers.Conv2D(filters=numfilters, kernel_size=kernel_size,
                                           strides=(1, 1), padding='same'),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.LeakyReLU()]

    def call(self, x):
        skip_layers = list()  # store skip layer outputs to be concatenated in the upsample block;
        for down_block, skip_block in zip(self.downsample_list, self.downsample_skip_block_list):
            for down_layer in down_block:  # traverse all layers in block;
                x = down_layer(x)
            if skip_block is not None:  # if there's a skip block, traverse all layers in it;
                x_ = x
                for skip_layer in skip_block:
                    x_ = skip_layer(x_)
                skip_layers.append(x_)
            else:
                skip_layers.append(None)
        for up_block, skip, concat in zip(self.upsample_list, skip_layers[::-1], self.upsample_concat_list):
            if skip is not None:
                x = concat([x, skip])
            for up_layer in up_block:
                x = up_layer(x)
        return x


# this function is used with the unet to calculate how to pad the input (and subsequently depad the output):
def get_compatible_size(dim, num_downsamp, max_dim=10000):
    # for a given dim and number of downsample blocks, find the smallest value >= dim such that the
    # unet will return the same size as the input;
    # max_dim is the largest value to be considered;
    k = 2 ** num_downsamp
    for i in range(1, max_dim):
        new_dim = i * k
        if new_dim >= dim:
            break
    return new_dim

# below are convenience functions for the script to call:
def stack_loader_phone(directory):
    # for my phone datasets;
    im_stack = list()
    for filename in tqdm(sorted(os.listdir(directory))):
        if 'jpg' in filename or 'png' in filename:
            im = plt.imread(directory + filename)
            im_stack.append(im)
    im_stack = np.array(im_stack)
    shape = im_stack.shape[1:]
    print('Shape of stack: ' + str(shape))
    return im_stack

def xcorr_initial_guess(im_stack, downsamp=4, channel=0, crop_frac=.4):
    # generates initial guess for x_s and y_pos using successive cross correlations (therefore, sequential images should
    # have overlap);
    # expect the input im_stack to be the output of an image loading script;
    # specify downsample factor (without antialiasing) and which channel to use;
    # crop_frac -- only search a center crop of the xcorr;

    im_stack_ = im_stack[:, ::downsamp, ::downsamp, channel].astype(np.float32)
    pos = [[0, 0]]  # first coordinate is arbitrarily defined to be the origin;
    for i in range(len(im_stack) - 1):
        xcorr = scipy.signal.correlate(im_stack_[i], im_stack_[i + 1])
        bkgd = scipy.signal.correlate(im_stack_[i], im_stack_[i])  # use autocorrelation as background;
        xcorr -= bkgd  # subtract background to avoid bias approaching 0 delay
        s = np.array(xcorr.shape)
        start_pix = np.int32((s - s * crop_frac) / 2)
        end_pix = np.int32((s + s * crop_frac) / 2)
        xcorr = xcorr[start_pix[0]:end_pix[0], start_pix[1]:end_pix[1]]

        argmax = np.unravel_index(xcorr.argmax(), xcorr.shape)
        pos.append(argmax - np.array(xcorr.shape) / 2)
    pos = np.array(pos).astype(np.float32) * downsamp
    pos = np.cumsum(pos, axis=0)  # convert relative shifts into absolute positions
    x_pos = pos[:, 1]
    y_pos = pos[:, 0]
    x_pos -= np.min(x_pos)  # avoid negative values because they will be stored as uints
    y_pos -= np.min(y_pos)

    return x_pos, y_pos

def monitor_progress(recon, error_map, losses, tracked=None):
    # run this in your train loop to print out progress;
    # recon is the reconstruction, error_map is the error map, losses is a list of MSEs or loss terms, and tracked
    # is a dictionary of tracked tensors from the tf graph;

    num_channels = recon.shape[-1]
    plt.figure(figsize=(15, 15))
    if num_channels == 1:
        plt.imshow(recon[:, :, 0].numpy(), cmap='jet')
        plt.colorbar()
    else:
        plt.imshow(np.uint8(np.clip(recon.numpy(), 0, 255)))
    plt.title('Reconstruction')
    plt.show()

    if error_map is not None:
        plt.figure(figsize=(15, 15))
        plt.imshow(error_map.numpy().sum(2))
        plt.clim([0, 300])
        plt.colorbar()
        plt.title('Error map')
        plt.show()

    if tracked is not None and 'height_map' in tracked:
        height_map = tracked['height_map']
        mask = recon.numpy().sum(2) > 0  # which pixels were visited (also could use normalize);
        clims = np.percentile(height_map[mask], [.1, 99.9])
        plt.figure(figsize=(15, 15))
        plt.imshow(height_map, cmap='jet')
        plt.clim(clims)
        plt.title('Height map')
        plt.colorbar()
        plt.show()

    plt.plot(losses)
    plt.title('Loss history')
    plt.show()
