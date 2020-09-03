# Polus Recycle Stitching Vector Plugin

This WIPP plugin applies an existing stitching vector to an image collection. Currently, the [WIPP image assembling plugin](https://github.com/usnistgov/WIPP-image-assembling-plugin) and the [WIPP pyramid building plugin](https://github.com/usnistgov/WIPP-pyramid-plugin) do not have a way to apply a stitching vector to new sets of images. For example, it is not currently possible to stitch the first channel of a multi-channel image and apply the same stitching vector to the other channels. This plugin creates a new stitching vector that will apply a single stitching vector to each channel in an image.

This plugin uses regular expressions similar to what MIST uses for filenames, but there are important exceptions that are described below in the section titled **Input Regular Expressions**.

For more information on WIPP, visit the [official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).

For more information on MIST, visit the [MIST repository](https://github.com/usnistgov/MIST).

## Input Regular Expressions
This plugin uses [filename patterns](https://github.com/USNISTGOV/MIST/wiki/User-Guide#input-parameters) similar to that of what MIST uses. In particular, defining a filename variable is surrounded by `{}`, and the variable name and number of spaces dedicated to the variable are denoted by repeated characters for the variable. For example, if all filenames follow the structure `filename_TTT.ome.tif`, where TTT indicates the timepoint the image was captured at, then the filename pattern would be `filename_{ttt}.ome.tif`.

All filename patterns must include `x` and `y` grid positions for each image or a sequential position `p`, but not both. This differs from MIST in that `r` and `c` are used to indicate grid row and column rather than `y` and `x` respectively. This change was made to remain consistent with Bioformats dimension names and to permit use of `c` as a channel variable.

In addition to the position variables (both `x` and `y`, or `p`), the only other variables that can be used are `r`, `z`, `c`, and `t`.

**Version 1.1 Update:** If no inputs are provided in the `groupBy` variable, then all non-positional variables that vary from the stitching vector will cause a new stitching vector to be generated. For example, if replicate, x, and y are all contained within a stitching vector but the vector contains only files for channel 1 (e.g. r{rrr}_x{xxx}_y{yyy}_c001.ome.tif), then the stitching vector filepattern can be set to r{rrr}_x{xxx}_y{yyy}_c{ccc}.ome.tif and `--groupBy r` can be used to recycle the stitching vector for additional channels.

## Build the plugin

To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## Install WIPP Plugin

In WIPP, navigate to the plugins page and add a new plugin. Paste the contents of `plugin.json` into the pop-up window and submit.

## Options

This plugin takes 5 input arguments and 1 output argument:

| Name                | Description                                            | I/O    | Type            |
|---------------------|--------------------------------------------------------|--------|-----------------|
| `--stitchDir`       | Stitching vector                                       | Input  | stitchingVector |
| `--collectionDir`   | Image collection                                       | Input  | collection      |
| `--stitchRegex`     | Regular expression for filenames in stitching vector   | Input  | String          |
| `--collectionRegex` | Regular expression for filenames in image collection   | Input  | String          |
| `--groupBy`         | String of variables that vary within a stiching vector | Input  | String          |
| `--outDir`          | Output stitching vector                                | Output | String          |

