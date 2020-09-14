# K-Means Clustering

The K-Means Clustering plugin clusters the data and outputs csv file.The input file should be in csv format.

## Building

To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin. Paste the contents of `plugin.json` into the pop-up window and submit.

## Options

This plugin takes eight input argument and one output argument:

| Name                   | Description             | I/O    | Type   |
|------------------------|-------------------------|--------|--------|
| `--inpdir` | Input csv file for clustering| Input | csvCollection |
| `--determinek` | Determine k-value automatically | Input | array |
| `--numofclus` | Enter k-value| Input | integer |
| `--outdir` | Output collection | Output | csvCollection |


