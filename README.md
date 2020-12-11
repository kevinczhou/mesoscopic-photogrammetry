# Mesoscopic photogrammetry with an unstabilized phone camera
This repository contains the code that stitches a sequence of close-range (several cms) camera images from different lateral positions of a scene with mm-scale height variation. The output is a stitched orthomosaic and a coaligned height map with accuracy as good as tens of microns. For more information on methodology and results, see our paper at TBD.

## Data
The image sequence datasets can be downloaded at TBD.

## Required packages
The code requires the following python 3 libraries:
- tensorflow 2.1\*
- numpy
- scipy
- cv2
- tqdm
- matplotlib
- jupyter

\*Please use IBM's version, which allows GPU-CPU memory swapping: https://github.com/IBM/tensorflow-large-model-support. Without this version, your GPU will probably run out of memory.

## Usage
First, download the datasets from the above figshare link, and put them in a directory, called `data/` (e.g., so that you have `data/painting/`, `data/helicopter_seeds/`, etc.). Next, run the jupyter notebook, `run_mesoscopic_photogrammetry.ipynb`, which contains more detailed, cell-by-cell instructions. To run as is, you will need a GPU with at least 11 GB of RAM and 32 GB of CPU RAM. With an Intel Xeon Silver 4116 augmented with an Nvidia RTX 2080 Ti GPU, we were able to perform the stitching and reconstruction in 8-12 hours per sample, if using the CNN reparameterization and without downsampling the images (1512x2016).
