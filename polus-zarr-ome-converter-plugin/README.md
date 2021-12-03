# Zarr OME  Converter (v0.1.0)

This WIPP plugin converts BioFormats supported data types to the Zarr OME file
format. 

## Building

To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin. Paste the
contents of `plugin.json` into the pop-up window and submit.

## Options

This plugin takes 1 input arguments and
1 output argument:

| Name            | Description                                                  | I/O    | Type        |
|-----------------|--------------------------------------------------------------|--------|-------------|
| `--inpDir`      | Input generic data collection to be processed by this plugin | Input  | genericData |
| `--filePattern` | A filepattern, used to select data for conversion            | Input  | genericData |
| `--outDir`      | Output collection                                            | Output | Output collection |

