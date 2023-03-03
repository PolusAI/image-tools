# Autocropping

This WIPP plugin automatically crops images to remove outer edges that do not contain useful information.
We do this by tracking the entropy of an edge as we move inward from that edge.
As soon as the entropy spikes, we know we have found tissue.
At this point, after adding some padding, we crop the image.

Note that entropy values are susceptible to structured noise.
It would be a good idea to apply noise removal techniques e.g., the flat-field correction, as a preprocessing step before trying to crop images with this plugin.

This plugin assumes that the input images have been assembled and stitched.

Contact [Najib Ishaq](mailto:najib.ishaq@axle.info) for more information.

For more information on WIPP, visit the [official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).

## Input Regular Expressions
This plugin uses [file patterns](https://filepattern.readthedocs.io/en/latest/Examples.html#what-is-filepattern) to group images in a collection.
In particular, we define a filename variable surrounded by `{}`, the variable name by a letter (see below), and number of spaces dedicated to the variable by repeating character for the variable.
For example, if all filenames follow the structure `filename_TTT.ome.tif`, where TTT indicates the time-point of capture, then the filename pattern would be `filename_{ttt}.ome.tif`.
If the number of repetitions are unknown or vary, `filename_{t+}.ome.tif` is also valid.

The available variables for filename patterns are `x`, `y`, `z`, `p`, `c` (channel), `t` (time-point), and `r` (replicate).
For position variables, only `x` and `y` grid positions or a sequential position `p` may be present, but not both.
This differs from MIST in that `r` and `c` are used to indicate grid row and column rather than `y` and `x` respectively.
We made this change to remain consistent with Bioformats dimension names and to permit use of `c` as a channel variable.

## Grouping Variables

The `groupBy` parameter can be used to define image groups using specific variables in their filename-patterns.
Each group is evaluated together and all images in the same group are cropped to the same bounding-box.

For example, if your images are named `r{r}_c{c}.ome.tif` where `r` represents imaging rounds and `c` represents channels, you can set `groupBy` to `c` to group together all images of different channels in the same round.
This will ensure that all images in the same round of imaging are cropped to the same bounding-box.
This will preserve any work done during tasks such as registration.

## Building

To build the Docker image for the conversion plugin, run `./build-docker.sh`.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin.
Paste the contents of `plugin.json` into the pop-up window and submit.

## Parameters

This plugin takes the following arguments:

| Name            | Description                                                                    | I/O    | Type       | Default |
| --------------- | ------------------------------------------------------------------------------ | ------ | ---------- | ------- |
| `--inputDir`    | Input collection.                                                              | Input  | collection | N/A     |
| `--filePattern` | File pattern to use for grouping images.                                       | Input  | string     | N/A     |
| `--groupBy`     | Grouping variables for images. Each group is cropped to the same bounding-box. | Input  | string     | N/A     |
| `--cropX`       | Whether to crop along the x-axis.                                              | Input  | boolean    | true    |
| `--cropY`       | Whether to crop along the y-axis.                                              | Input  | boolean    | true    |
| `--cropZ`       | Whether to crop along the z-axis.                                              | Input  | boolean    | true    |
| `--smoothing`   | Whether to use gaussian smoothing on images to add more tolerance to noise.    | Input  | boolean    | true    |
| `--outputDir`   | Output collection.                                                             | Output | collection | N/A     |
