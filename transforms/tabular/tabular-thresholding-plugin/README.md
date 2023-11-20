# Tabular Thresholding Plugin (v0.1.3)
This plugin uses three [threshold methods](https://github.com/nishaq503/thresholding.git) to compute threshold values on a user-defined variable and then determines if each label (ROI) is above or below the calculated threshold value. A new feature column will be computed for selected threshold method with the values in  binary format (0, 1) \
*0* `negative or below threshold`\
*1* `positive or above threshold`

## Threshold methods

### *1-* False Positive Rate
It estimates mean and standard deviation of `negControl` values based on the assumption that it follows a single guassian distribution and computes threshold such that the area to the right is equal to a user-defined `falsePositiverate`. Values must range between 0 and 1

### *2-* OTSU
It computes threshold by using `negControl` and `posControl` values to minimize the weighted variance of these two classes. `numBins` are number of bins to compute histogram of `negControl` and `posControl` values

### *3-* MEAN+Sigma
It computes threshold by calculating mean and `n` number of standard deviations of `negControl` values.

Contact [Hamdah Shafqat Abbasi](mailto: hamdah.abbasi@axleinfo.com) for more information.

For more information on WIPP, visit the [official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).

## Building

To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin. Paste the
contents of `plugin.json` into the pop-up window and submit.

## Options

This plugin takes 10 input arguments and one output argument:

| Name                    | Description                                         | I/O    | Type          |
|-------------------------|-----------------------------------------------------|--------|---------------|
| `--inpDir`              | Input directory containing tabular data CSVs        | Input  | genericData   |
| `--filePattern`         | Pattern to parse tabular files                      | Input  | string        |
| `--negControl`          | FeatureName describing non treated wells/ROI        | Input  | string        |
| `--posControl`          | FeatureName describing treated wells/ROI            | Input  | string        |
| `--varName`             | FeatureName for thresholding                        | Input  | string        |
| `--thresholdType`       | See above in README                                 | Input  | enum          |
| `--falsePositiverate`   | Area to the right of the threshold                  | Input  | float         |
| `--numBins`             | Number of bins for histogram                        | Input  | number        |
| `--n`                   | Number of standard deviation                        | Input  | number        |
| `--outFormat`           | Output file format                                  | Input  | enum          |
| `--outDir`              | Output collection                                   | Output | genericData   |
