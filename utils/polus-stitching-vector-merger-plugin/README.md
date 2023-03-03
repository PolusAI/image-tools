# Polus Stitching Vector Collection Merger Plugin

This WIPP plugin merges stitching vector collections together. It takes as input a minimum of 2 collections upto a maximum of 5 collections.

 Contact [Gauhar Bains](mailto:gauhar.bains@labshare.org) for more information.

For more information on WIPP, visit the [official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).

## Building

To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin. Paste the contents of `plugin.json` into the pop-up window and submit.

## Options

This plugin takes one input argument and one output argument:

| Name                  | Description                    | I/O    | Type            |
| --------------------- | ------------------------------ | ------ | --------------- |
| `--VectorCollection1` | 1st stitchingVector Collection | Input  | stitchingVector |
| `--VectorCollection2` | 2nd stitchingVector Collection | Input  | stitchingVector |
| `--VectorCollection3` | 3rd stitchingVector Collection | Input  | stitchingVector |
| `--VectorCollection4` | 4th stitchingVector Collection | Input  | stitchingVector |
| `--VectorCollection5` | 5th stitchingVector Collection | Input  | stitchingVector |
| `--outDir`            | Output collection              | Output | stitchingVector |
