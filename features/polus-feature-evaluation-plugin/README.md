# Feature Evaluation Plugin

Plugin to generate evaluation metrics for feature comparison of ground truth and predicted images. Contact [Vishakha Goyal](mailto:vishakha.goyal@nih.gov) for more information.

For more information on WIPP, visit the [official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).

## Building

To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin. Paste the contents of `plugin.json` into the pop-up window and submit.

## Options

This plugin takes four input arguments and one output argument:

| Name          | Description             | I/O    | Type   |
|---------------|-------------------------|--------|--------|
| `--GTDir` | Ground truth feature collection to be processed by this plugin. | Input | csvCollection |
| `--PredDir` | Predicted feature collection to be processed by this plugin. | Input | csvCollection |
| `--combineLabels` | Boolean to calculate number of bins for histogram by combining GT and Predicted Labels. Default is using GT labels only. | Input | boolean |
| `--outFileFormat` | Boolean to save output file as csv. Default is lz4. | Input | boolean |
| `--singleCSV` | Boolean to save output file as a single csv. Default is true. | Input | boolean |
| `--filePattern` | Filename pattern to filter data. | Input | string |
| `--outDir` | Output collection | Output | genericData |