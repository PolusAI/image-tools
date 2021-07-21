# Label to Vector  plugin

This Wipp plugin converts a labelled image to vector field and stores the vector field as an ome zarr.
A vector field consists of a cell-probability value along with x and y components for 2d images or x, y and z components for 3d images.
Each component (x, y, z, cell probability) is stored as a separate channel, and channel labels are stored in the metadata for reference.
If no `filePattern` is supplied, all the images in a directory are processed.

## A Note on 3d Flow Fields

Cellpose was originally built to predict flow fields for 2d images.
The authors extended this to 3d images by aggregating the 2d-flows for each slice in the image,
i.e. every z-slice, every y-slice and every x-slice.
This makes the flow-field calculations for 3d images considerably slow.
The ideal solution for this is to build and train a Cellpose-like model to directly predict 3d flows.
This would necessitate the use of 3d Convolutional layers instead of 2d layers.

## Building

To build the Docker image for the conversion plugin, run `./build-docker.sh`.

## Using Container Locally

To run the container, type 

An example of how to run the plugin is included in the shell script, `run-plugin.sh`.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin.
Paste the contents of `plugin.json` into the pop-up window and submit.

## Options

This plugin takes two input arguments and one output argument:

| Name            | Description                                                   | I/O    | Type         |
|-----------------|---------------------------------------------------------------|--------|--------------|
| `--inpDir`      | Input image collection to be processed by this plugin.        | Input  | collection   |
| `--filePattern` | Input image file name pattern to be processed by this plugin. | Input  | string       |
| `--outDir`      | Output collection.                                            | Output | Generic data |

