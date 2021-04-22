# WIPP Widget

This WIPP plugin trains and tests a neural network with SplineDist in order to automate cell segmentation with Spline Interpolation. 

This plugin requires the user to specify how to split the testing and training data:
1) Specify the percentage that the input directories are split into
2) Specify the testing directories where the images are located

The performance is then evaluated using the Jaccard Index.
A plot is created with 3 subplots showing the original image, ground truth, and the predicted image.

For more information on SplineDist:
Paper: https://www.biorxiv.org/content/10.1101/2020.10.27.357640v1
Github Repository: https://github.com/uhlmanngroup/splinedist

Contact [Madhuri Vihani](madhuri.vihani@axleinfo.com) for more information.

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
| `--inpImageDirTest` | Path to folder with intesity based images for testing | Input | string |
| `--inpLabelDirTest` | Path to folder with labelled segments, or ground truth, for testing | Input | string |
| `--outDir` | Path to where model gets saved to | Output | string |
| `--splitPercentile` | Percentage of data that is allocated for testing from training directories | Input | int |
| `--gpuAvailability` | Specifies whether or not a GPU is available to use for training | Input | bool |
| `--action` | Specifies whether or not the plugin is testing or training | Input | string |
| `--controlPoints` | The number of control that are used to define the shape of the segments | Input | int |
| `--epochs` | Number of epochs to run to train neural network | Input | string |
| `--learningRate` | Specifies the learning rate if it needs to be updated to continue training | Input | string |
| `--imagePattern` | Filename pattern used to separate data | Output | string |

