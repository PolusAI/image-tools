# MESMER Training(v0.0.6-dev)

This WIPP Plugin trains PanopticNet using MESMER Pipeline.

Paper -
Whole-cell segmentation of tissue images with human-level performance using large-scale data annotation and deep learning
Noah F. Greenwald, Geneva Miller, Erick Moen, Alex Kong, Adam Kagel, Christine Camacho Fullaway, Brianna J. McIntosh, Ke Leow, Morgan Sarah Schwartz, Thomas Dougherty, Cole Pavelchek, Sunny Cui, Isabella Camplisson, Omer Bar-Tal, Jaiveer Singh, Mara Fong, Gautam Chaudhry, Zion Abraham, Jackson Moseley, Shiri Warshawsky, Erin Soon, Shirley Greenbaum, Tyler Risom, Travis Hollmann, Leeat Keren, Will Graf, Michael Angelo, David Van Valen
bioRxiv 2021.03.01.431313; doi: https://doi.org/10.1101/2021.03.01.431313

API reference -

https://deepcell.readthedocs.io/en/master/API/deepcell.html

https://deepcell.readthedocs.io/en/master/notebooks/Training-Segmentation.html

Contact [Vishakha Goyal](mailto:vishakha.goyal@nih.gov) , [Hamdah Shafqat Abbasi](mailto:hamdahshafqat.abbasi@nih.gov) for more information.

For more information on WIPP, visit the [official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).

## Building

To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin. Paste the contents of `plugin.json` into the pop-up window and submit.

## Options

This plugin takes ten input arguments and one output argument:

| Name          | Description             | I/O    | Type   |
|---------------|-------------------------|--------|--------|
| `--trainingImages` | Input training image collection to be processed by this plugin | Input | collection |
| `--trainingLabels` | Input training mask collection to be processed by this plugin | Input | collection |
| `--testingImages` | Input testing image collection to be processed by this plugin | Input | collection |
| `--testingLabels` | Input testing mask collection to be processed by this plugin | Input | collection |
| `--modelBackbone` | Keras models to be use as backbone for Deepcell model | Input | enum |
| `--filePattern` | Pattern to parse images | Input | string |
| `--tilesize` | Tile size to be processed by the model. Default is 256. | Input | number |
| `--iterations` | Number of training iterations. Default is 10. | Input | number |
| `--batchSize` | Batch size for training. Default is 1. | Input | number |
| `--outDir` | Output collection | Output | genericData |
| `--preview` | Generate JSON file with outputs | Output | JSON |
