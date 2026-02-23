# BaSiC Flatfield Correction (v2.1.2-dev0)

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

The `basicpy` package does not yet support a calculation of photobleach, so this plugin can only calculate the flatfield and darkfield images.

## Input Regular Expressions

This plugin uses [filepattern](https://filepattern.readthedocs.io/en/latest/Examples.html#what-is-filepattern) to select images from the input collection.
In particular, defining a filename variable is surrounded by `{}`, and the variable name and number of spaces dedicated to the variable are denoted by repeated characters for the variable.
For example, if all filenames follow the structure `filename_TTT.ome.tif`, where TTT indicates the timepoint the image was captured at, then the filename pattern would be `filename_{t:ddd}.ome.tif`.

## Build the plugin

To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## Install WIPP Plugin

In WIPP, navigate to the plugins page and add a new plugin. Paste the contents
of `plugin.json` into the pop-up window and submit.

## Running tests

To run the tests, the command is `pytest`. This will run all tests in the `tests` directory.

Since this plugin is only a thin wrapper around the `basicpy` package, the tests are limited to verifying that the plugin is able to run and that the output images are generated.
The tests do not verify that the output images are correct.
This check is performed by the `basicpy` package, which has its own tests.

## Options

This plugin takes 4 input arguments and 1 output argument:

| Name            | Description                                                 | I/O    | Type    |
|-----------------|-------------------------------------------------------------|--------|---------|
| `--inpDir`      | Path to input images                                        | Input  | String  |
| `--darkfield`   | If 'true', will calculate darkfield image                   | Input  | Boolean |
| `--photobleach` | If 'true', will calculate photobleach scalar                | Input  | Boolean |
| `--filePattern` | File pattern to subset data                                 | Input  | String  |
| `--groupBy`     | Variables to group together                                 | Input  | String  |
| `--outDir`      | Output image collection                                     | Output | String  |
| `--preview`     | Preview the names of output images without running any code | Input | Boolean |
