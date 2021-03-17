# WIPP Widget

This WIPP plugin does Morphological Image Processing on binary images.  
The operations available are: 

  * Invertion
  * Dilation
  * Erosion
  * Opening
  * Closing
  * Morphological Gradient
  * Filling Holes
  * Skeletonization
  * Top Hat
  * Black Hat


Contact [Data Scientist](mailto:Madhuri.Vihani@axleinfo.com) for more information.
For more information on WIPP, visit the [official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).

## Building

To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin. Paste the contents of `plugin.json` into the pop-up window and submit.

## Options

This plugin takes five input arguments and has output argument:

| Name          | Description             | I/O    | Type   |
|---------------|-------------------------|--------|--------|
| `--inpDir` | Input image collection to be processed by this plugin | Input | collection |
| `--outDir` | Output collection | Output | collection |
| `--Operation`| The Morphological Operation to be done on input images | Input | String |
| `--structuringshape`| Shape of the structuring element can either be Elliptical, Rectangular, or Cross | Input | String |
| `--kernelsize`| Size of the kernel for most operations | Input | String |


