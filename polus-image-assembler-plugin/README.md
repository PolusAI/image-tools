# Image Assembler

This WIPP plugin assembles images into a stitched image using an image stitching vector. The need for this plugin is due to limitations of the NIST [WIPP Image Assembling Plugin](https://github.com/usnistgov/WIPP-image-assembling-plugin) that limits image sizes to less than 2^32 pixels. The NIST plugin works faster since it is built on optimized C-code. If your image has less than 2^32 pixels, it is recommended to use NISTs image assembling algorithm.

For more information on WIPP, visit the [official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).

## Building

To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin. Paste the contents of `plugin.json` into the pop-up window and submit.

## Options

This plugin takes two input arguments and one output argument:

| Name          | Description             | I/O    | Type   |
|---------------|-------------------------|--------|--------|
| `--stitchPath` | Path to stitching vector | Input | stitchingVector |
| `--imgPath` | Path to input image collection | Input | collection |
| `--outDir` | Path to output image collection | Output | collection |

