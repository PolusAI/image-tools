# Outlier removal (v0.2.7-dev0)

The outlier removal plugin removes the outliers from the data based on the method selected and outputs csv file. The output will have separate csv files for inliers and outliers. The input file should be in csv format.

The plugin support vaex supported input csv file that need outliers to be removed. The file should be in csv format. This is a required parameter for the plugin.

## Methods

Choose any one of the methods mentioned to remove outliers from the data.

### Isolation Forest

Ensemble-based unsupervised method for outlier detection. The algorithm isolates outliers instead of normal instances. It works based on the principle that outliers are few and different and hence, the outliers can be identified easier than the normal points. The score is calculated as the path length to isolate the observation. These two methods can be selected to detect outliers>

1. `IsolationForest` Detect outliers globally that deviates significantly from the rest of the datapoints
2. `IForest` Detect local outliers that are distinct when compared to those of its neighbors.


### Global

<img src="images/Global.PNG" width="500" height="500">

### Local

<img src="images/Local.PNG" width="500" height="500">

## Outputs:

Select the output file by passing value to `outputType`. User can select from following options `inlier`, `oulier` or `combined`. The combined file contains `anomaly` column which score each datapoint if it is inlier or outlier.

## Building

To build the Docker image for the conversion plugin, run
`./build-docker.sh`

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin. Paste the contents of `plugin.json` into the pop-up window and submit.
For more information on WIPP, visit the [official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).

## Options

This plugin takes three input arguments and one output argument:

| Name        | Description                           | I/O    | Type          |
| ----------- | ------------------------------------- | ------ | ------------- |
| `--inpDir`  | Input directory containing tabular files | Input  | genericData   |
| `--filePattern`  | Pattern to parse tabular file names                  | Input  | string   |
| `--methods` | Select methods for outlier removal    | Input  | enum          |
| `--outputType`   | Select type of output file | Input  | enum          |
| `--outdir`  | Output collection                     | Output | genericData   |
| `--preview`        | Generate a JSON file with outputs                                  | Output | JSON          |
