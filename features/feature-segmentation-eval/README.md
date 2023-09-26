# Feature segmentation eval (v0.2.3)

Plugin to generate evaluation metrics for feature comparison of ground truth and predicted images. Contact [Vishakha Goyal](mailto:vishakha.goyal@nih.gov), [Hamdah Shafqat Abbasi](mailto:hamdahshafqat.abbasi@nih.gov) for more information.

For more information on WIPP, visit the [official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).

## Building

To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin. Paste the contents of `plugin.json` into the pop-up window and submit.

## Options

This plugin takes six input arguments and one output argument:

| Name          | Description             | I/O    | Type   |
|---------------|-------------------------|--------|--------|
| `--GTDir` | Ground truth feature collection to be processed by this plugin. | Input | genericData |
| `--PredDir` | Predicted feature collection to be processed by this plugin. | Input | genericData |
| `--filePattern` | Filename pattern to filter data. | Input | string |
| `--combineLabels` &nbsp; | Boolean to calculate number of bins for histogram by combining GT and Predicted Labels | Input | boolean |
| `--singleOutFile` &nbsp; | Boolean to save output file as a single file.| Input | boolean |
| `--outDir` | Output collection | Output | genericData |
| `--preview`           | Generate a JSON file with outputs                            | Output | JSON        |
