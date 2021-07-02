# Autocropping

This WIPP plugin automatically crops images to remove outer rows/columns that do not contain useful information.
We do this by tracking the entropy of the rows/columns as we move inward from an edge.
As soon as the entropy spikes, we know we have found tissue.
At this point, after adding some padding, we crop the image.

Note that entropy values are susceptible to structured noise.
It would be a good idea to apply noise removal techniques e.g., the flat-field correction plugin, as a preprocessing step before trying to crop images with this plugin.

Contact [Najib Ishaq](mailto:najib.ishaq@axle.info) for more information.

For more information on WIPP, visit the [official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).

## Input Regular Expressions
This plugin uses [filepatterns](https://filepattern.readthedocs.io/en/latest/Examples.html#what-is-filepattern) to group images in a collection.
In particular, we define a filename variable surrounded by `{}`, the variable name by a letter (see below), and number of spaces dedicated to the variable by repeating character for the variable.
For example, if all filenames follow the structure `filename_TTT.ome.tif`, where TTT indicates the timepoint of capture, then the filename pattern would be `filename_{ttt}.ome.tif`.
If the number of repetitions are unknown or vary, `filename_{t+}.ome.tif` is also valid.

The available variables for filename patterns are `x`, `y`, `z`, `p`, `c` (channel), `t` (timepoint), and `r` (replicate).
For position variables, only `x` and `y` grid positions or a sequential position `p` may be present, but not both.
This differs from MIST in that `r` and `c` are used to indicate grid row and column rather than `y` and `x` respectively.
We made this change to remain consistent with Bioformats dimension names and to permit use of `c` as a channel variable.

## Building

To build the Docker image for the conversion plugin, run `./build-docker.sh`.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin.
Paste the contents of `plugin.json` into the pop-up window and submit.

## Options

This plugin takes 1 input arguments and 1 output argument:

| Name          | Description             | I/O    | Type   | Default Values |
|---------------|-------------------------|--------|--------|------|
| `--inputDir` | Input collection. | Input | collection |  |
| `--filePattern` | File pattern to use for grouping images. | Input | string |  |
| `--groupBy` | Which file-pattern variables to use for grouping images. | Input | str |  |
| `--axes` | Whether to crop rows, columns or both. | Input | enum | both |
| `--smoothing` | Whether to use gaussian smoothing on images to add more tolerance to noise. | Input | boolean | true |
| `--outputDir` | Output collection. | Output | collection |  |
