# Vector to label plugin
This plugin turns
[Cellpose](https://www.biorxiv.org/content/10.1101/2020.02.02.931238v1)
flow fields into a labeled image. 
The plugin takes a vector field as input and generates masks based on the flow
error and cell probability threshold entered by user.

A meshgrid is generated based on pixel location and pixels are grouped based on
where they converge. These grouped pixels form a mask. From the masks flows are
recomputed and masks above the flow error threshold are removed.

The author's recommended values for cell probability threshold, flow error, and
stitching threshold are 0, 0.4, and 0.

See the
[original paper](https://www.biorxiv.org/content/10.1101/2020.02.02.931238v1)
or the
[Cellpose github repository](https://github.com/MouseLand/cellpose/tree/master/cellpose)
for more information.

## Building

To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## To run container locally
The `run-plugin.sh` script has an example of how to run the plugin locally.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin. Paste the
contents of `plugin.json` into the pop-up window and submit.

## Options

This plugin takes 4 input arguments and 1 output argument:

| Name                  | Description                | I/O    | Type        | Default values |
|-----------------------|----------------------------|--------|-------------|----------------|
| `--inpDir`            | Input image collection     | Input  | GenericData | n/a            |
| `--flowThreshold`     | flow threshold             | Input  | number      | 0.8            |
| `--cellprobThreshold` | Cell probability threshold | Input  | number      | 0              |
| `--outDir`            | Output Path                | Output | collection  | n/a            |

