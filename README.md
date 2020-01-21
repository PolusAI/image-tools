# Polus Plugins - Deep Learning

WIPP Plugin Title : Cell Nuclei Segmentation using U-net

Credits for the Neural network and model weigths : https://github.com/axium/Data-Science-Bowl-2018/

For more information on WIPP, visit the [official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).

For more information on Bioformats, vist the [official page](https://www.openmicroscopy.org/bio-formats/).

## Description 
This WIPP plugin segments cell nuclei using U-Net in Tensorflow. Neural net architecture and pretrained weights are taken from Data Science Bowl 2018 entry by Muhammad Asim. The unet expects the input height and width to be 256 pixels. To ensure that the plugin is able to handle images of all sizes, it adds reflective padding to the input to make the dimensions a multiple of 256. Following this a loop extracts 256x256 tiles to be processed by the network. In the end it untiles and removes padding from the output. 

The plugin takes 2 inputs as shown below :\
(i) Path to the input directory - The directory should consist of  grayscale images to be segmented.\
(ii) Path to the output directory. The output is a binary mask highlighting the nuclei. 

## Options 
This plugin takes one input argument and one output argument:

| Name       | Description             | I/O    | Type |
|------------|-------------------------|--------|------|
| `inpDir`   | Input image collection  | Input  | Path |
| `outDir`   | Output image collection | Output | Path |






