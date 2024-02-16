# Theia Bleedthrough Estimation (v0.5.2-dev0)

This WIPP plugin estimates the bleed-through in a collection of 2d images.
It uses the Theia algorithm from [this repo](https://github.com/PolusAI/theia).

## File Patterns

This plugin uses [file-patterns](https://filepattern.readthedocs.io/en/latest/Examples.html#what-is-filepattern) to create subsets of an input collection.
In particular, defining a filename variable is surrounded by `{}`, and the variable name and number of spaces dedicated to the variable are denoted by repeated characters for the variable.
For example, if all filenames follow the structure `prefix_tTTT.ome.tif`, where `TTT` indicates the time-point of capture of the image, then the file-pattern would be `prefix_t{ttt}.ome.tif`.

## Optional Parameters

### --groupBy

This parameter can be used to group images into subsets.
This plugin will apply bleed-through correction to each subset.
Each subset should contain all channels for one image/tile/FOV.
The images in each subset should all have the same size (in pixels and dimensions) and one.

If no `--groupBy` is specified, then the plugin will assume that all images in the input collection are part of the same subset.

### --selectionCriterion

Which method to use to rank and select tiles in images.
The available options are:

1. `"MeanIntensity"`: Select tiles with the highest mean pixel intensity. This is the default.
2. `"Entropy"`: Select tiles with the highest entropy.
3. `"MedianIntensity"`: Select tiles with the highest median pixel intensity.
4. `"IntensityRange"`: Select tiles with the largest difference in intensity of the brightest and dimmest pixels.

We rank-order all tiles based on one of these criteria and then select some of the best few tiles from each channel.
If the images are small enough, we select all tiles from each channel.

### --channelOrdering

By default, we assumed that the order of channel numbers is the same as the order, in increasing wavelength, of the emission filters for those channels.
If this is not the case, use this parameter to specify, as a string of comma-separated integers, the wavelength-order of the channels.

For example, if the channels are `0, 1, 2, 3, 4` and they correspond to wavelengths (of the emission filter) of `420nm, 350nm, 600nm, 510nm, 580nm`, then `--channelOrdering` should be `"1,0,3,4,2"`.

If this parameter is not specified, then we assume that the channel numbers are in increasing wavelength order.

If you do not know the channel ordering, you can use the `--channelOverlap` parameter to specify a higher number of adjacent channels to consider as contributors to bleed-through.

### --channelOverlap

For each channel in the image, we assume that the only noticeable bleed-through is from a small number of adjacent channels.
By default, we consider only `1` adjacent channel on each side of the wavelength scale as contributors to bleed-through.

For example, for channel 3, we would consider channels 2 and 4 to contribute bleed-through components.

Use a higher value for `--channelOverlap` to have our model look for bleed-through from farther channels.

### --kernelSize

We learn a convolutional kernel for estimating the bleed-through from each channel to each neighboring channel.
This parameter specifies the size of those kernels.

We recommend one of `3`, `5`, or `7` and use `3` as the default.

## TODOs:

1. Handle case where each image file contains all channels.
2. Extend to 3d images.

## Build the plugin

To build the Docker image for the conversion plugin, run `./build-docker.sh`.

## Install WIPP Plugin

In WIPP, navigate to the plugins page and add a new plugin.
Paste the contents of `plugin.json` into the pop-up window and submit.

## Options

This plugin takes 6 input arguments and 1 output argument:

| Name                   | Description                              | I/O    | Type    | Default         |
| ---------------------- | ---------------------------------------- | ------ | ------- | --------------- |
| `--inpDir`             | Input image collection.                  | Input  | String  | N/A             |
| `--filePattern`        | File pattern to subset images.           | Input  | String  | ".*"            |
| `--groupBy`            | Variables to group together.             | Input  | String  | ""              |
| `--channelOrdering`    | Channel ordering by wavelength scale.    | Input  | String  | ""              |
| `--selectionCriterion` | Method to use for selecting tiles.       | Input  | Enum    | "MeanIntensity" |
| `--channelOverlap`     | Number of adjacent channels to consider. | Input  | Integer | 1               |
| `--kernelSize`         | Size of convolutional kernels to learn.  | Input  | Integer | 3               |
| `--outDir`             | Output image collection.                 | Output | String  | N/A             |
