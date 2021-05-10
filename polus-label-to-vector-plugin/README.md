# Label to Vector  plugin 
This Wipp plugin converts a labelled image to vector field. Vector field is concatenation of the horizontal,vertical gradients and cell probability. 


The output will be a zarr file . Each image folder in the zarr file will have it's associated vector field and label since the label is required 
for Cellpose segmentation training plugin. This plugin has been tested on bfio version 2.0.4-slim-buster.

## Building
To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## Using Container Locally
`docker run   -v {label image collection}:/opt/executables/input -v {output zarr Dir}:/opt/executables/output 
labshare/polus-label-to-vector-plugin:{version}  --inpDir /opt/executables/input --outDir /opt/executables/output` 

## Install WIPP Plugin
If WIPP is running, navigate to the plugins page and add a new plugin. Paste the contents of `plugin.json` into the 
pop-up window and submit.

## Options
This plugin takes one input argument and one output argument:

| Name          | Description             | I/O    | Type   |
|---------------|-------------------------|--------|--------|
| `--inpDir` | Input image collection to be processed by this plugin | Input | collection |
| `--inpRegex` | Input image file name pattern to be processed by this plugin | Input | string |
| `--outDir` | Output collection | Output | Generic data |

