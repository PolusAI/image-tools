# Intensity Projection Plugin

This WIPP plugin calculates the volumeteric intensity projection of a 3d image along the z-direction(depth).  The following types of intensity projections have been implemented: 

1. Maximum: 
2. Minimum 
3. Mean 
```
Example: Consider an input image of size: (x,y,z). If the user chooses the option `max`, the code will calculate.  
the value of the maximum intensity value along the z-direction for every x,y position. The output will be a 2d   
image of size (x,y). 
```
Contact [Gauhar Bains](mailto:gauhar.bains@labshare.org) for more information.

For more information on WIPP, visit the [official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).

## Building

To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin. Paste the contents of `plugin.json` into the pop-up window and submit.

## Options

This plugin takes one input argument and one output argument:

| Name          | Description             | I/O    | Type   |
|---------------|-------------------------|--------|--------|
| `--inpDir` | Input image collection to be processed by this plugin | Input | collection |
| `--projectionType` | Type of volumetric intensity projection | Input | string |
| `--outDir` | Output collection | Output | collection |

