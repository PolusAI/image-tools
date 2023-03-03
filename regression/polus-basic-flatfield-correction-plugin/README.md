# BaSiC Flatfield Correction

This WIPP plugin will take a collection of images and use the BaSiC flatfield
correction algorithm to generate a flatfield image, a darkfield image, and a
photobleach offset. BaSiC was originally described by Peng et al in
[A BaSiC tool for background and shading correction of optical microscopy images](https://doi.org/10.1038/ncomms14836).

This Python implementation was developed in part using a Matlab repository that
appears to have been created by one of the authors on
[github](https://github.com/QSCD/BaSiC). This implementation is numerically
similar (but not identical) to the Matlab implementation, where similar means
the difference in the results is <0.1%. This appears to be the result of
differences between Matlab and OpenCV's implementation of bilinear image
resizing.

An image pattern field is provided for the WIPP input that permits subdividing
the data by z-slice, timepoint, and channel, so that separate flatfield,
darkfield, and photobleaching offsets can be generated for each subset of
images.

## Input Regular Expressions
This plugin uses
[filepatterns](https://filepattern.readthedocs.io/en/latest/Examples.html#what-is-filepattern)
to select data in an input collection for .
In particular, defining a filename variable is surrounded by `{}`, and the
variable name and number of spaces dedicated to the variable are denoted by
repeated characters for the variable. For example, if all filenames follow the
structure `filename_TTT.ome.tif`, where TTT indicates the timepoint the image
was captured at, then the filename pattern would be `filename_{ttt}.ome.tif`.

The available variables for filename patterns are `x`, `y`, `p`, `z`, `c`
(channel), `t` (timepoint), and `r` (replicate). For position variables, only
`x` and `y` grid positions or a sequential position `p` may be present, but not
both. This differs from MIST in that `r` and `c` are used to indicate grid row
and column rather than `y` and `x` respectively. This change was made to remain
consistent with Bioformats dimension names and to permit use of `c` as a channel
variable.

## Build the plugin

To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## Install WIPP Plugin

In WIPP, navigate to the plugins page and add a new plugin. Paste the contents
of `plugin.json` into the pop-up window and submit.

## Options

This plugin takes 4 input arguments and 1 output argument:

| Name            | Description                                  | I/O    | Type    |
| --------------- | -------------------------------------------- | ------ | ------- |
| `--inpDir`      | Path to input images                         | Input  | String  |
| `--darkfield`   | If 'true', will calculate darkfield image    | Input  | Boolean |
| `--photobleach` | If 'true', will calculate photobleach scalar | Input  | Boolean |
| `--filePattern` | File pattern to subset data                  | Input  | String  |
| `--groupBy`     | Variables to group together                  | Input  | String  |
| `--outDir`      | Output image collection                      | Output | String  |
