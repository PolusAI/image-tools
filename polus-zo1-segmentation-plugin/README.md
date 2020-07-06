# ZO1 Border Segmentation

This plugin segments cell borders when fluorescently labeled for ZO1 tight junction protein. A neural network is used for segmentation, and it was trained on cells retinal pigment epithelial cells from multiple organisms, from multiple labs, different microscopes, and at multiple magnifications. The neural network used in this plugin was originally reported in the publication ["Deep learning predicts function of live retinal pigment epithelium from quantitative microscopy"](https://www.jci.org/articles/view/131187).

For more information on WIPP, visit the [official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).

## Building

To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin. Paste the contents of `plugin.json` into the pop-up window and submit.

## Options

This plugin takes one input argument and one output argument:

| Name       | Description                                           | I/O    | Type       |
|------------|-------------------------------------------------------|--------|------------|
| `--inpDir` | Input image collection to be processed by this plugin | Input  | collection |
| `--outDir` | Output collection                                     | Output | collection |

