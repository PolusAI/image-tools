# Polus Image Registration Plugin

WIPP Plugin Title : Image Registration Plugin 

Contact [Gauhar Bains](mailto:gauhar.bains@labshare.org) for more information.

For more information on WIPP, visit the [official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).

## Description

This plugin registers an image collection. First it 
 It uses projective transformation(Homography) to transform the moving image and align it with the reference image. 

Background information about homography can be found here: https://en.wikipedia.org/wiki/Homography

## Algorithm 



 Initially, the plugin parses the input collection and divides it into sets. Each set consists of 3 things 

## Building

To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin. Paste the contents of `plugin.json` into the pop-up window and submit.

## Options

This plugin takes one input argument and one output argument:

| Name          | Description             | I/O    | Type   |
|---------------|-------------------------|--------|--------|
| `--filePattern` | Filename pattern used to separate data | Input | string |
| `--inpDir` | Input image collection to be processed by this plugin | Input | collection |
| `--registrationVariable` | variable to help identify which images need to be registered to each other | Input | string |
| `--template` | Template image to be used for image registration | Input | string |
| `--TransformationVariable` | variable to help identify which images have similar transformation | Input | string |
| `--outDir` | Output collection | Output | collection |

