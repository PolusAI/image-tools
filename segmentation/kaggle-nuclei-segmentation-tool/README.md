# Kaggle Nuclei Segmentation (v0.1.5-dev2)

Segments cell nuclei using U-Net in Tensorflow.

# Reference
Credits neural network architecture and pretrained model weights : https://github.com/axium/Data-Science-Bowl-2018

# Description
This WIPP plugin segments cell nuclei using U-Net in Tensorflow. Neural net architecture and pretrained weights are taken from Data Science Bowl 2018 entry by Muhammad Asim (reference given above). The unet expects the input height and width to be 256 pixels. To ensure that the plugin is able to handle images of all sizes, it adds reflective padding to the input to make the dimensions a multiple of 256. Following this a loop extracts 256x256 tiles to be processed by the network. In the end it untiles and removes padding from the output.

## Building

To build the Docker image for the conversion plugin, run `./build-docker.sh`.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin. Paste the
contents of `plugin.json` into the pop-up window and submit.

## Options

This plugin takes 3 input arguments and 1 output argument:

| Name          | Description             | I/O    | Type   | Default |
|---------------|-------------------------|--------|--------|--------- |
| `--inpDir`       | Input image collection to be processed by this plugin | Input | collection |
| `--filePattern`  | Filename pattern used to separate data | Input | string | .* |
| `--outDir`       | Output collection | Output | collection |
| `--preview`  | Generate an output preview | Input | boolean | False |
