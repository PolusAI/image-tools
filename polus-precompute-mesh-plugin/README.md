# WIPP Widget

This WIPP plugin handles 3D images and creates the necessary files to be compatible with Neuroglancer. 

Two types of inputs can be handled:

1) Intensity Based Images
2) Segmentation
    This type of data can also generate meshes with Levels of Details.  

Contact [Madhuri Vihani](Madhuri.Vihani@axleinfo.com) for more information.

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
| `--inpDir` | Input image collection to be processed by this plugin | Input | collection |
| `--imageType` | Input is either an image or labeled data set (segmentation) | Input | String |
| `--meshes` | Generate Meshes | Input | Boolean |
| `--outDir` | Output collection | Output | collection |

