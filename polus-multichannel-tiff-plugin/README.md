# Multichannel Tiff

The multi-channel tiff plugin uses a 
[filename pattern](https://github.com/LabShare/polus-plugins/tree/master/utils/polus-filepattern-util)
to assign images to a multi-channel ome tiff. Only channels indicated in the
`channelOrder` are included in the multi-channel tiff, and channels are placed
in the order that they are input. For example, if `channelOrder = 1,5,3`, then
only files with channel variables 1, 5, and 3 will be merged into the file and
will be placed into channesl 1, 2, and 3 respectively.

For more information on WIPP, visit the
[official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).

## Building

To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin. Paste the
contents of `plugin.json` into the pop-up window and submit.

## Options

This plugin takes one input argument and one output argument:

| Name             | Description                            | I/O    | Type        |
|------------------|----------------------------------------|--------|-------------|
| `--filePattern`  | Filename pattern used to separate data | Input  | string      |
| `--inpDir`       | Input image collection                 | Input  | collection  |
| `--channelOrder` | What channel to assign each image to   | Input  | array       |
| `--outDir`       | Output collection                      | Output | genericData |

