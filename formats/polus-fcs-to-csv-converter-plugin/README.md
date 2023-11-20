# Fcs to Csv file converter

The fcs to csv file converter plugin converts fcs file to csv file.The input file should be in .fcs file format and output will be .csv file format.

## Input:
The input should be a file in fcs format.

## Output:
The output is a csv file.

For more information on WIPP, visit the [official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).

## Building

To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin. Paste the contents of `plugin.json` into the pop-up window and submit.

## Options

This plugin takes eight input argument and one output argument:

| Name       | Description               | I/O    | Type          |
| ---------- | ------------------------- | ------ | ------------- |
| `--inpDir` | Input fcs file collection | Input  | collection    |
| `--outDir` | Output collection         | Output | csvCollection |


