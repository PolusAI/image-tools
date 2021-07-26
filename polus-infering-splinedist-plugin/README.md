# SplineDist Inference Plugin

This WIPP plugin uses a trained SplineDist Model to make predictions on intensity based images
This WIPP plugin is also capable of making predictions in tiles

For more information on SplineDist:
[Published Paper](https://www.biorxiv.org/content/10.1101/2020.10.27.357640v1)  
[Github Repository](https://github.com/uhlmanngroup/splinedist)  
SideNote: The input images are filled.  There are no empty holes in the output predictions made with SplineDist

Contact [Madhuri Vihani](madhuri.vihani@axleinfo.com) for more information.

For more information on WIPP, visit the [official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).

## Building

To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin. Paste the contents of `plugin.json` into the pop-up window and submit.

## Options

This plugin takes one input argument and one output argument:

| Name             | Description                                                      | I/O    | Type   |
|------------------|------------------------------------------------------------------|--------|--------|
| `--inpImageDir`  | Path to folder with intesity based images to make predictions on | Input  | string |
| `--inpBaseDir`   | Path to folder containing Splinedist Model's weights             | Input  | string |
| `--imagePattern` | Filename pattern used to separate data                           | Input  | string |
| `--outDir`       | Path to where output labelled images get saved to                | Output | string |
