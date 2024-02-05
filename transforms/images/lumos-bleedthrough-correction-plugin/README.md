# LUMoS Bleedthrough Correction Plugin (v0.1.1-dev0)

This WIPP plugin will take a collection of images and use the LUMoS [1] bleedthrough correction algorithm to separate out the signal for each fluorophore.
This is a reimplementation, in Python, of the authors' original [Java](https://github.com/tristan-mcrae-rochester/Multiphoton-Image-Analysis/blob/master/Spectral%20Unmixing/Code/ImageJ-FIJI/LUMoS_Spectral_Unmixing.java) and [Matlab](https://github.com/tristan-mcrae-rochester/Multiphoton-Image-Analysis/blob/master/Spectral%20Unmixing/Code/k_means_unmixing_circ/KMeansUnmixing.m) versions.
The algorithm assumes that fluorophores are spatially separated in the images and uses k-means clustering to separate the individual signals in the input image.

This plugin expects input images in the `.ome.tif` or `.ome.zarr` format.
The output will be multiple single-channel images in the `.ome.zarr` format.

If each input image file contains a multichannel image, the `groupBy` parameter should be the empty string.
Otherwise, if each input image file contains a single-channel image, then the `groupBy` input parameter must be used to group together all channels of the same multichannel image.

There is no requirement for the number of channels to be equal to the number of fluorophores.
Instead, this plugin expects an input integer `numFluorophores` to denote the number of fluorophores in the multichannel images.
The output will be `n + 1` single-channel images, where `n` is the number of fluorophores.
There will be one channel for background and each of the other channels will contain signal for each individual fluorophore.

Note that the LUMoS algorithm itself makes no guarantee for the order of the output channels, including which channel is the background channel.
There is an effort made in this plugin that the last channel is the background signal, but it is not an absolute guarantee.

## Input Regular Expressions

This plugin uses [filepattern](https://filepattern.readthedocs.io/en/latest/Examples.html#what-is-filepattern) to select data in an input collection.
In particular, defining a filename variable is surrounded by `{}`, and the variable name and number of spaces dedicated to the variable are denoted by repeated characters for the variable.
For example, if all filenames follow the structure `filename_TTT.ome.tif`, where TTT is a three digit integer indicating the time-point at which the image was captured, then the filename pattern would be `filename_{t:ddd}.ome.tif`.

## Build the plugin

To build the Docker image for the conversion plugin, run `./build-docker.sh`.

## Install WIPP Plugin

In WIPP, navigate to the plugins page and add a new plugin.
Paste the contents of `plugin.json` into the pop-up window and submit.

## Inputs

This plugin takes 4 input arguments and 1 output argument:

| Name                | Description                          | I/O    | Type    |
|---------------------|--------------------------------------|--------|---------|
| `--inpDir`          | Path to input images                 | Input  | String  |
| `--filePattern`     | Filepattern for input images.        | Input  | String  |
| `--groupBy`         | Grouping variables for input images. | Input  | String  |
| `--numFluorophores` | Number of fluorophores in images.    | Input  | Integer |
| `--outDir`          | Output image collection              | Output | String  |

## Citations

[1] McRae TD, Oleksyn D, Miller J, Gao Y-R (2019) Robust blind spectral unmixing for fluorescence microscopy using unsupervised learning. PLoS ONE 14(12): e0225410. https://doi.org/10.1371/journal.pone.0225410
