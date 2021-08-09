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

This plugin is designed to make the flow field calculations scalable, operating
on images too large to fit into memory. It does this by converting flows to
labels in 2048x2048 pixel tiles at a time with 256 pixel overlap. This plugin
also only uses 200 iterations when following flow fields. This means that
objects with a radius roughly larger than 200 pixels may not be reconstructed
properly. In the future, it may be necessary to add tile size, overlap, and
number of iterations as inputs. However, for most of the image types that
CellPose is properly suited for segmenting, these settings should cover the
majority of images that will be segmented.

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

