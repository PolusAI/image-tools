# Image Assembler

This WIPP plugin assembles images into a stitched image using an image stitching vector. The need for this plugin is due to limitations of the NIST [WIPP Image Assembling Plugin](https://github.com/usnistgov/WIPP-image-assembling-plugin) that limits image sizes to less than 2^32 pixels. The NIST plugin works faster since it is built on optimized C-code. If your image has less than 2^32 pixels, it is recommended to use NISTs image assembling algorithm.

For more information on WIPP, visit the [official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).

## To Do

Currently the plugin does not have a good way of designating output image names. It searches all image characters for every image to find differences in image names and assigns ranges of values in the output image name. For example, if an image collection has the following set of images that will be assembled:

```bash
x001_y001.ome.tif
x001_y002.ome.tif
x002_y001.ome.tif
x002_y002.ome.tif
```

Then the output image name will be `x00<0-1>_y00<0-1>.ome.tif`. However, if the images range from `000` to `010` trailing the x and y positions of the image names, then the output image name will be `x0<00-19>_y00<00-19>.ome.tif` because the 2nd digit ranges from 0-1 and the 3rd digit ranges from 0-9. However, this could give the impression that the resulting image contains a 20x20 grid of images where the x and y grid coordinates range from `000` to `019`. A better way of naming output images should be developed.

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
