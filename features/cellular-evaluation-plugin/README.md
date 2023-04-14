# Cellular Evaluation Plugin(v0.2.2)

Plugin to generate evaluation metrics for region-level comparison of ground truth and predicted images. Contact [Vishakha Goyal](mailto:vishakha.goyal@nih.gov) , [Hamdah Shafqat Abbasi](mailto:hamdahshafqat.abbasi@nih.gov) for more information.

For more information on WIPP, visit the [official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).

## Building

To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin. Paste the contents of `plugin.json` into the pop-up window and submit.

## Options

This plugin takes eleven input arguments and one output argument:

| Name                  | Description             | I/O    | Type   |
|-----------------------|-------------------------|--------|--------|
| `--GTDir`             | Ground truth input image collection to be processed by this plugin | Input | collection |
| `--PredDir`           | Predicted input image collection to be processed by this plugin | Input | collection |
| `--inputClasses`      | Number of classes | Input | number |
| `--individualData`    | Boolean to calculate individual image statistics. Default is false. | Input | boolean |
| `--individualSummary` | Boolean to calculate summary of individual images. Default is false. | Input | boolean |
| `--totalStats`        | Boolean to calculate overall statistics across all images. Default is false. | Input | boolean |
| `--totalSummary`      | Boolean to calculate summary across all images. Default is false. | Input | boolean |
| `--radiusFactor`      | Importance of radius/diameter to find centroid distance. Should be between (0,2]. Default is 0.5.| Input | string |
| `--iouScore`          | IoU theshold. Default is 0.| Input | string |
| `--filePattern`       | Filename pattern to filter data. | Input | string |
| `--fileExtension`     | A desired file format for tabular outputs | Input  | enum        |
| `--preview`           | Generate a JSON file with outputs                            | Output | JSON        |
