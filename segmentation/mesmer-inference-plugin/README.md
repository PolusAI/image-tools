# MESMER Inference

This WIPP plugin segments images using PanopticNet model. 

Paper - 
Whole-cell segmentation of tissue images with human-level performance using large-scale data annotation and deep learning
Noah F. Greenwald, Geneva Miller, Erick Moen, Alex Kong, Adam Kagel, Christine Camacho Fullaway, Brianna J. McIntosh, Ke Leow, Morgan Sarah Schwartz, Thomas Dougherty, Cole Pavelchek, Sunny Cui, Isabella Camplisson, Omer Bar-Tal, Jaiveer Singh, Mara Fong, Gautam Chaudhry, Zion Abraham, Jackson Moseley, Shiri Warshawsky, Erin Soon, Shirley Greenbaum, Tyler Risom, Travis Hollmann, Leeat Keren, Will Graf, Michael Angelo, David Van Valen
bioRxiv 2021.03.01.431313; doi: https://doi.org/10.1101/2021.03.01.431313

API reference - 

https://deepcell.readthedocs.io/en/master/API/deepcell.html 

https://deepcell.readthedocs.io/en/master/notebooks/Training-Segmentation.html

Contact [Vishakha Goyal](mailto:vishakha.goyal@nih.gov) for more information.

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
| `--testingImages` | Intensity images to be processed by this plugin. | Input | collection |
| `--testingLabels` | Label images to be processed by this plugin. Optional. | Input | collection |
| `--tilesize` | Tile size to be processed by the model. Default is 256. Optional. | Input | number |
| `--modelPath` | Model path for inference. Optional. | Input | genericData |
| `--model` | Model - mesmerNuclear, mesmerWholeCell, BYOM | Input | enum |
| `--filePatternTest` | Filename pattern for test data. | Input | string |
| `--filePatternWholeCell` | Filename pattern for nuclear images for whole cell segmentation. Optional.| Input | string |
| `--outDir` | Output collection | Output | collection |

