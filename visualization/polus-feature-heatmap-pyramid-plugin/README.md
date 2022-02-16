# Feature Heatmap Pyramid

This WIPP plugin uses an existing image collection, stitching vector, and csv collection containing image features and generates an image collection containing heatmap values and a stitching vector to assemble the heatmap images into a pyramid using the [WIPP Pyramid Plugin](https://github.com/usnistgov/WIPP-pyramid-plugin). Currently there isn't any way to associate the stitching vector to a feature.

One heatmap pyramid is generated for each feature (csv column) in the feature csv file. The pyramids are stored as timeframes for viewing in WDZT. Currently there is no way to assign labels or tags to each timeframe, so determining which timeframe is associated with a feature is challenging.

For more information on WIPP, visit the [official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).

## Building

To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin. Paste the contents of `plugin.json` into the pop-up window and submit.

## Options

This plugin takes three input arguments and two output arguments:

| Name          | Description             | I/O    | Type   |
|---------------|-------------------------|--------|--------|
| `--features` | CSV collection containing features | Input | csvCollection |
| `--inpDir` | Input image collection used to build a pyramid that this plugin will make an overlay for | Input | collection |
| `--vector` | Stitching vector used to buld the image pyramid. | Input | stitchingVector |
| `--outImages` | Output image directory for heatmap images | Output | collection |
| `--outVectors` | Output image directory for heatmap vectors | Output | stitchingVector |
