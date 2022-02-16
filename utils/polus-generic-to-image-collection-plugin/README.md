# Generic to Image Collection

This WIPP plugin copies the contents of a Generic Data collection to an Image
Collection. This is useful if the original data uploaded is uploaded as a
Generic Data collection and is already in the tiled OME TIFF file spec. The
data in the Generic Data collection is checked for the proper file extension
(`.ome.tif`) and for the proper tile size (tile width and height = 1024)
before copying. Only data with the proper file extension are copied, and an
error will be thrown if a file does not have the appropriate tile size.

For more information on WIPP, visit the
[official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).

## Building

To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin. Paste the
contents of `plugin.json` into the pop-up window and submit.

## Options

This plugin takes 1 input arguments and
1 output argument:

| Name          | Description             | I/O    | Type   |
|---------------|-------------------------|--------|--------|
| `--inpDir` | Input image collection to be processed by this plugin | Input | genericData |
| `--outDir` | Output collection | Output | collection |
