# Polus Recycle Stitching Vector Plugin

This WIPP plugin applies an existing stitching vector to an image collection.
The
[WIPP image assembling plugin](https://github.com/usnistgov/WIPP-image-assembling-plugin)
and the
[WIPP pyramid building plugin](https://github.com/usnistgov/WIPP-pyramid-plugin)
do not have a way to apply a stitching vector to new sets of images. For
example, it is currently possible to stitch the first channel of a multi-channel
image and apply the same stitching vector to the other channels. This plugin
creates a new stitching vector that will apply an existing single stitching
vector to a different set of images.

This plugin uses 
[filepatterns](https://filepattern.readthedocs.io/en/latest/),
which is a variant of regular expressions similar to what MIST uses. See
**Input Regular Expressions** for more information.

## Input Regular Expressions
A `filepattern` is a way of indicating where variables in a file name are
located. A `filepattern` defines a filename variable using `{}`, and the
variable name and number of spaces dedicated to the variable are denoted by
repeated characters for the variable. For example, if all filenames follow the
structure `filename_TTT.ome.tif`, where TTT indicates the timepoint the image
was captured at, then the filename pattern would be `filename_{ttt}.ome.tif` or
`filename_{t+}.ome.tif`. For more information on `filepattern`, see the 
[documentation](https://filepattern.readthedocs.io/en/latest/).

## Build the plugin

To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## Install WIPP Plugin

In WIPP, navigate to the plugins page and add a new plugin. Paste the contents
of `plugin.json` into the pop-up window and submit.

## Options

This plugin takes 5 input arguments and 1 output argument:

| Name                | Description                                            | I/O    | Type            |
|---------------------|--------------------------------------------------------|--------|-----------------|
| `--stitchDir`       | Stitching vector                                       | Input  | stitchingVector |
| `--collectionDir`   | Image collection                                       | Input  | collection      |
| `--stitchRegex`     | `filepattern` for filenames in stitching vector        | Input  | String          |
| `--collectionRegex` | `filepattern` for filenames in image collection        | Input  | String          |
| `--groupBy`         | String of variables that vary within a stiching vector | Input  | String          |
| `--outDir`          | Output stitching vector                                | Output | stitchingVector |

### stitchRegex and collectionRegex

The `stitchRegex` and `collectionRegex` must contain either `x` and `y`
variables, or a `p` variable. These variables are what the plugin use to match
the location of corresponding files. If any other variables are provided, then
a new stitching vector is created for each unique value extracted for that
variable.

### groupBy

The `groupBy` variables permits grouping variables into the same stitching
vector. This is useful when the filenames in the stitching vector do not match
the files in the image collection and there may be more than one set of values
in the filename that change.