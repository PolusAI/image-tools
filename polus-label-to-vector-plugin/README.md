# Label-to-Vector  plugin 
This plugin converts a labelled image to vector field.The plugin stores the vector field  as well as the label since the 
label is required for the training plugin. It's opposite  of what vector-label plugin does. This plugin is 
also starting point for cellpose segmentation training plugin.

To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin. Paste the contents of `plugin.json` into the pop-up window and submit.

## Options

This plugin takes one input argument and one output argument:

| Name          | Description             | I/O    | Type   |
|---------------|-------------------------|--------|--------|
| `--inpDir` | Input image collection to be processed by this plugin | Input | collection |
| `--outDir` | Output collection | Output | Generic data |

