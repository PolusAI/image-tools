# Feature Extraction

This WIPP plugin does things, some of which involve math and science. There is likely a lot of handwaving involved when describing how it works, but handwaving should be replaced with a good description. However, someone forgot to edit the README, so handwaving will have to do for now. Contact [Jayapriya Nagarajan](mailto:jayapriya.nagarajan@labshare.org) for more information.

For more information on WIPP, visit the [official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).

## Building

To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin. Paste the contents of `plugin.json` into the pop-up window and submit.

## Options

This plugin takes one input argument and one output argument:

| Name          | Description             | I/O    | Type   |
|---------------|-------------------------|--------|--------|
| `--angleDegree` | Angle degree to calculate feret diameter | Input | integer |
| `--boxSize` | Boxsize to calculate feret diameter | Input | integer |
| `--intDir` | Intensity image collection| Input | collection |
| `--pixelDistance` | Pixel distance to calculate the neighbors touching cells | Input | integer |
| `--segDir` | Segment image collection | Input | collection |
| `--features` | Select intensity and shape features required | Input | array |
| `--csvfile` | Save csv file as one csv file for all images or separate csv file for each image | Input | array |
| `--outDir` | Output collection | Output | collection |

