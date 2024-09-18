# OME Converter (v0.3.3-dev1)

This WIPP plugin converts BioFormats supported data types to the
OME Zarr or OME TIF file format. This is not a complete implementation, rather it implements a file
format similar to the OME tiled tiff specification used by WIPP. Chunk sizes
are 1024x1024x1x1x1, and OME metadata is stored as a Zarr attribute.

For more information on the OME Zarr format, read the
[OME NGFF file specification](https://ngff.openmicroscopy.org/latest/).

For more information on WIPP, visit the
[official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).

## Building

To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin. Paste the
contents of `plugin.json` into the pop-up window and submit.

## Options

This plugin takes 3 input arguments and 1 output argument:

| Name             | Description                                                  | I/O    | Type        |
|------------------|--------------------------------------------------------------|--------|-------------|
| `--inpDir`       | Input generic data collection to be processed by this plugin | Input  | genericData |
| `--filePattern`  | A filepattern, used to select data for conversion            | Input  | string      |
| `--fileExtension`| A desired file format for conversion                         | Input  | enum        |
| `--outDir`       | Output collection                                            | Output | genericData |
| `--preview`      | Generate a JSON file with outputs                            | Output | JSON        |
