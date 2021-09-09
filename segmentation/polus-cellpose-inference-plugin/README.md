# Cellpose Inference Plugin

[Cellpose](https://www.biorxiv.org/content/10.1101/2020.02.02.931238v1) is a generalist algorithm for cell and nucleus segmentation.
Cellpose uses two major innovations: a reversible transformation from training set masks to vector flows that can be predicted by a neural network, and a large segmented dataset of varied images of cells.

This plugin is an implementation of segmenting cyto/nuclei in 2D and/or 3D images using pretrained models created by authors of Cellpose.
Apart from allowing users to specify the type of segmentation, this plugin also allows the users to choose the diameter of cells as well as providing the option to input custom pretrained model.
If you are using a custom pretrained model, make sure to have it compatible with the CellposeModel class as defined in [`cellpose.models`](https://cellpose.readthedocs.io/en/latest/api.html#cellposemodel).

When choosing the diameter and inputting custom model, keep in mind that:
1. Default diameter for cyto and nuclei models is `17` and `30` respectively.
   If `0` is passed as diameter plugin will estimate diameter for each image.
   However, diameter estimation is not available for 3d images because cellpose implemented no supporting methods.
   If you have 3d images then you will likely need to provide a default `diameter` to use.
2. The option to estimate diameter for each image will not be available if when using a custom model.
   In this case, the user must specify the model diameter to be used.

This plugin predicts flow-field vectors and cell-probabilities.
There are saved as a zarr array for each input image.
This plugin has been tested with `CUDA 11.1 - 11.4`, `pytorch 1.9.0` and `bfio 2.1.9`.
It runs on GPU(s) by default.

## Building

To build the Docker image for the conversion plugin, run `./build-docker.sh`.

## Steps to reproduce Docker container locally

1. Pull the container:

`docker pull labshare/polus-cellpose-inference-plugin:${version}`

2. Run the container:

`docker run -v {inpDir}:/opt/executables/input -v {outDir}:/opt/executables/output labshare/polus-cellpose-inference-plugin:{version} --inpDir /opt/executables/input --outDir /opt/executables/output`

Add `--gpus {device no}` as an argument to use gpu in container.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin.
Paste the contents of `plugin.json` into the pop-up window and click submit.

## Options

This plugin takes 5 input arguments and 1 output argument:

| Name | Description | I/O | Type |
|------|-------------|-----|------|
| `--inpDir` | Input image collection to be processed by this plugin. | Input | collection |
| `--filePattern` | File pattern for selecting files to segment. | Input | string |
| `--diameter` | Estimated diameter of objects in the images. | Input | number |
| `--diameterMode` | Method of setting diameter. | Input | enum |
| `--pretrainedModel` | Name of builtin model or path to custom model. | Input | string |
| `--outDir` | Output collection. | Output | collection |
