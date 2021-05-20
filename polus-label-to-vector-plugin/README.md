# Label to Vector  plugin

This Wipp plugin converts a labelled image to vector field and stores the vector field as a zarr.
Vector field is concatenation of the horizontal,vertical gradients and cell probability. Vector
field and label are stored as zarr groups.

The plugin lets users filter images in the input image collection based on file names. By
default, the plugin will run on the all the images in image collection. Plugin has been tested on
bfio version 2.0.4-slim-buster and serves as a starting point for Cellpose training plugin.

## Building

To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## Using Container Locally

To run the container, type 

`docker run -v {label image collection}:/opt/executables/input -v {output zarr Dir}:/opt/executables/output labshare/polus-label-to-vector-plugin:{version} --inpDir /opt/executables/input --outDir /opt/executables/output`

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin. Paste the contents
of `plugin.json` into the pop-up window and submit.

## Options

This plugin takes one input argument and one output argument:

| Name          | Description             | I/O    | Type   |
|---------------|-------------------------|--------|--------|
| `--inpDir` | Input image collection to be processed by this plugin | Input | collection |
| `--inpRegex` | Input image file name pattern to be processed by this plugin | Input | string |
| `--outDir` | Output collection | Output | Generic data |

