# Pytorch Inference Plugin

This plugin performs inference, i.e., image segmentation, on arbitrarily large ome.tif images using any pytorch neural network that was saved using torchscript.

## Compatible Models

This plugin makes some basic assumptions about the model being loaded.
Please make sure that your model adheres to this set of requirements.

1. The model must be a subclass of `torch.nn.Module` and must have been saved using torchscript: `torch.jit.script(model).save("path/to/model.pth")`. The file must be named "model.pth". Torchscript will save the parameters and weights of the trained model so no additional files are required.
2. The model must have a `forward` method that takes a batch of single-channel input images and produces output images with the same spatial dimensions as the input images. Any preprocessing on the images must be done in the `forward` method. For scalability, this plugin performs inference on tiles of shape 1024x1024 so the input shape to the model must be a divisor of of this tile shape.
3. The input should be two-dimensional, single-channel images, though the output may have multiple channels.
4. The model should use the `float32` data type.

## Device Selection

This plugin can (optionally) use multiple GPUs for parallelized inference.
This can be configured using the `device` input parameter.
The allowed values are:

- "cpu": The plugin will run inference on the CPU.
- "gpu": The plugin will use a single GPU.
- "all": The plugin will use every available GPU.
- A comma-separated string of integers, e.g. "1,3,6", in which case, the plugin will use the indexed GPUs.

## Constraints on Input and Output Images 

The inputs should be two-dimensional, single-channel ome.tif images.
These images may be arbitrarily large in the spatial dimensions.

Since the model may produce images that have multiple channels, the output images from this plugin will be in the ome.zarr format.
They will have the same metadata as the corresponding input images though they will have been saved using `numpy.float32` as the data type.

## Building

To build the Docker image for the conversion plugin, run `./build-docker.sh`.

## Steps to reproduce Docker container locally

1. Pull the container:

`docker pull polusai/pytorch-inference-plugin:0.1.0`

2. Run the container:

See the `run-plugin.sh` script for an example. 

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin.
Paste the contents of `plugin.json` into the pop-up window and click submit.

## Options

This plugin takes 5 input arguments and 1 output argument:

| Name            | Description                                                          | I/O    | Type         |
|-----------------|----------------------------------------------------------------------|--------|--------------|
| `--modelDir`    | Directory with a "model.pth" file which can be used to load a model. | Input  | Generic Data |
| `--imagesDir`   | Input image collection to be processed by this plugin.               | Input  | collection   |
| `--filePattern` | File pattern for selecting files to segment.                         | Input  | string       |
| `--device`      | Which device(s) to use for running the model.                        | Input  | string       |
| `--outDir`      | Output collection.                                                   | Output | collection   |
