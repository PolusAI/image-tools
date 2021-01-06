# Outlier removal
The outlier removal plugin removes the outliers from the data using Isolation Forest algorithm and outputs csv file. The output is separate csv files for inliers and outliers. The input file should be in csv format.

## Inputs:
### Input csv collection:
The input file that need outliers to be removed. The file should be in csv format. This is a required parameter for the plugin.

## Methods:
Choose any one of the method mentioned to remove outliers from the data.

### Isolation Forest
Ensemble-based unsupervised method for outlier detection. The algorithm isolates outliers instead of normal instances. It works based on the principle that outliers are few and different and hence, the outliers can be identified easier than the normal points. The score is calculated as the path length to isolate the observation.

## Output:
The output is separate csv files for inliers and outliers.

## Building

To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin. Paste the contents of `plugin.json` into the pop-up window and submit.
For more information on WIPP, visit the [official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).

## Options

This plugin takes four input argument if methods other than 'Manual' is selected else three input arguments and one output argument:

| Name                   | Description             | I/O    | Type   |
|------------------------|-------------------------|--------|--------|
| `--inpdir` | Input csv collection| Input | csvCollection |
| `--methods` | Select methods for outlier removal | Input | enum|
| `--outdir` | Output collection | Output | csvCollection |