# Outlier removal
The outlier removal plugin removes the outliers from the data based on the method selected and outputs csv file. The output will have separate csv files for inliers and outliers. The input file should be in csv format.

## Inputs:
### Input csv collection:
The input csv file that need outliers to be removed. The file should be in csv format. This is a required parameter for the plugin.

## Methods:
Choose any one of the methods mentioned to remove outliers from the data.

### Isolation Forest
Ensemble-based unsupervised method for outlier detection. The algorithm isolates outliers instead of normal instances. It works based on the principle that outliers are few and different and hence, the outliers can be identified easier than the normal points. The score is calculated as the path length to isolate the observation.

## Types:
Choose any one of the types(global/local) mentioned to remove outliers from the data. This is an optional parameter. Based on the method selected for detection of outliers, this option is displayed.

### Global
Select global to remove outliers that deviates significantly from the rest of the datapoints.
<img src="images/Global.PNG" width="500" height="500">

### Local
Select local to remove outliers that are distinct when compared to those of its neighbors.
<img src="images/Local.PNG" width="500" height="500">

## Output:
Separate csv files for inliers and outliers.

## Building

To build the Docker image for the conversion plugin, run
`./build-docker.sh`

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin. Paste the contents of `plugin.json` into the pop-up window and submit.
For more information on WIPP, visit the [official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).

## Options

This plugin takes three input arguments and one output argument:

| Name                   | Description             | I/O    | Type   |
|------------------------|-------------------------|--------|--------|
| `--inpdir` | Input csv collection| Input | csvCollection |
| `--methods` | Select methods for outlier removal | Input | enum|
| `--types` | Select type of outliers to be removed| Input | enum|
| `--outdir` | Output collection | Output | csvCollection |
