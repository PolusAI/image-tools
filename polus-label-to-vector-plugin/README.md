# Label-to-Vector  plugin 
This plugin converts a labelled image to vector field.The plugin stores the vector field  as well as the label since the 
label is required for the training plugin. It's opposite  of what vector-label plugin does. This plugin is 
also starting point for cellpose segmentation training plugin.
This plugin  iterates through slices of masks in a labelled array  and computes the horizontal and vertical gradients.
The gradients are computed by starting a diffusion process  at the center of masks. Plugin saves the horizontal,
vertical gradients  and cell probability in  chunks as well as labelled image itself. Both the arrays are inputs 
required for cellpose training plugin.

This plugin has been tested on bfio version 2.0.4-slim-buster.


## Building
To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## Using Container Locally
  `docker run   -v {label image collection}:/opt/executables/input -v {output zarr Dir}:/opt/executables/output labshare/polus-label-to-vector-plugin:{version}  --inpDir /opt/executables/input --outDir /opt/executables/output` 

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin. Paste the contents of `plugin.json` into the 
pop-up window and submit.


## Options

This plugin takes one input argument and one output argument:

| Name          | Description             | I/O    | Type   |
|---------------|-------------------------|--------|--------|
| `--inpDir` | Input image collection to be processed by this plugin | Input | collection |
| `--outDir` | Output collection | Output | Generic data |

