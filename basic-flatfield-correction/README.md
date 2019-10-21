# BaSiC Flatfield Correction

This WIPP plugin will take a collection of images and use the BaSiC flatfield correction algorithm to generate a flatfield image, a darkfield image, and a photobleach offset. BaSiC was originally described by Peng et al in [A BaSiC tool for background and shading correction of optical microscopy images](https://doi.org/10.1038/ncomms14836).

This Python implementation was developed in part using a Matlab repository that appears to have been created by one of the authors on [github](https://github.com/QSCD/BaSiC). This implementation is numerically similar (but not identical) to the Matlab implementation, where similar means the difference in the results is <0.1%. This appears to be the result of differences between Matlab and OpenCV's implementation of bilinear image resizing.

An image pattern field is provided for the WIPP input that permits subdividing the data by z-slice, timepoint, and channel, so that separate flatfield, darkfield, and photobleaching offsets can be generated for each subset of images.

## Input Regular Expressions
This plugin uses [filename patterns](https://github.com/USNISTGOV/MIST/wiki/User-Guide#input-parameters) similar to that of what MIST uses. In particular, defining a filename variable is surrounded by `{}`, and the variable name and number of spaces dedicated to the variable are denoted by repeated characters for the variable. For example, if all filenames follow the structure `filename_TTT.ome.tif`, where TTT indicates the timepoint the image was captured at, then the filename pattern would be `filename_{ttt}.ome.tif`.

All filename patterns must include `x` and `y` grid positions for each image or a sequential position `p`, but not both. This differs from MIST in that `r` and `c` are used to indicate grid row and column rather than `y` and `x` respectively. This change was made to remain consistent with Bioformats dimension names and to permit use of `c` as a channel variable.

In addition to the position variables (both `x` and `y`, or `p`), the only other variables that can be used are `z`, `c`, and `t`. Images with the same `z`, `t`, and `c` will be grouped to calculate a flatfield, darkfield, and photobleach offset.

## Known Issues

This plugin was tested on a reasonably large collection (18,500 images with 55 subsets). As the plugin cycled through each image subset, loading images became progressively slower (from about 0.3 seconds to 3 seconds to load each image).

## Build the plugin

To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## Install WIPP Plugin

In WIPP, navigate to the plugins page and add a new plugin. Paste the contents of `plugin.json` into the pop-up window and submit.

## Options

This plugin takes 4 input arguments and 1 output argument:

| Name            | Description                                  | I/O    | Type   |
|-----------------|----------------------------------------------|--------|--------|
| `--inpDir`      | Path to input images                         | Input  | String |
| `--darkfield`   | If 'true', will calculate darkfield image    | Input  | String |
| `--photobleach` | If 'true', will calculate photobleach scalar | Input  | String |
| `--inpRegex`    | File pattern to subset data                  | Input  | String |
| `--outDir`      | Output image collection                      | Output | String |

