# SplineDist Training Plugin

This WIPP plugin trains and tests a neural network with SplineDist in order to automate cell segmentation with Spline Interpolation. 

This plugin requires the user to specify how to split the testing and training data:
The user can either:
1) Specify the percentage that the input directories are split into
2) Specify the testing directories where the images are located
In order to specify the testing directories then split percentile must be either 0 or None.

The user must also specify the number of Control Points, which defines the number of points for closed planar spline curve that generates the mask on one cyst.
More details are described in SplineDist's paper.

For more information on SplineDist:
[Published Paper](https://www.biorxiv.org/content/10.1101/2020.10.27.357640v1)
[Github Repository](https://github.com/uhlmanngroup/splinedist)
SideNote: The input images are filled.  There are no empty holes in the output predictions made with SplineDist

Contact [Madhuri Vihani](madhuri.vihani@nih.gov) for more information.

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
| `--inpImageDirTrain` | Path to folder with intesity based images for training | Input | string |
| `--inpLabelDirTrain` | Path to folder with labelled segments, or ground truth, for training | Input | string |
| `--outDir` | Path to where model gets saved to | Output | string |
| `--splitPercentile` | Percentage of data that is allocated for testing from training directories | Input | int |
| `--controlPoints` | The number of control that are used to define the shape of the segments | Input | int |
| `--epochs` | Number of epochs to run to train neural network | Input | string |
| `--imagePattern` | Filename pattern used to separate data | Output | string |