# Bleed-Through Estimation

This WIPP plugin estimates the bleed-through in a collection of 2d images.
We started with the algorithm described in [this paper](https://doi.org/10.1038/s41467-021-21735-x) and implemented in [this repo](https://github.com/RoysamLab/whole_brain_analysis).
During the course of our work on this plugin, we realized and implemented several improvements.
We will link the paper here once it is published.

This plugin requires Pyhton 3.9+ along with the packages listed in `requirements.txt`.

The `filePattern` and `groupBy` parameters can be used to group images into subsets.
This plugin will apply bleed-through correction to each subset.
Each subset should contain all channels for one image/tile/FOV.

## File Patterns

This plugin uses [file-patterns](https://filepattern.readthedocs.io/en/latest/Examples.html#what-is-filepattern) to create subsets of an input collection.
In particular, defining a filename variable is surrounded by `{}`, and the variable name and number of spaces dedicated to the variable are denoted by repeated characters for the variable.
For example, if all filenames follow the structure `prefix_tTTT.ome.tif`, where `TTT` indicates the time-point of capture of the image, then the file-pattern would be `prefix_t{ttt}.ome.tif`.

The available variables for filename patterns are `x`, `y`, `p`, `z`, `c` (channel), `t` (time-point), and `r` (replicate).
For position variables, either `x` and `y` grid positions or a sequential position `p` may be present, but not both.
This differs from MIST in that `r` and `c` are used to indicate grid row and column rather than `y` and `x` respectively.
This change was made to remain consistent with Bioformats dimension names and to permit the use of `c` as a channel variable.

## Optional Parameters

### --groupBy

Each round of staining should be its own group.
This suggests that `'r'` and `'c'` should be among the grouping variables.
You can also use `'x'`, `'y'`, `'z'` and/or `'p'` to further subgroup tiles if the fields-of-view have not yet been stitched together.

### --selectionCriterion

Which method to use to rank and select tiles in images.
The available options are:

* `'MeanIntensity'`: Select tiles with the highest mean pixel intensity. This is the default.
* `'Entropy'`: Select tiles with the highest entropy.
* `'MedianIntensity'`: Select tiles with the highest median pixel intensity.
* `'IntensityRange'`: Select tiles with the largest difference in intensity of the brightest and dimmest pixels.

We rank-order all tiles based on one of these criteria and then select the 10 best tiles from each channel.
From the selected tiles, we then select the brightest pixels to create the training data presented to the model.
A pixel chosen for any channel is used for all channels.

### --model

Which model (from `sklearn.linear_model`) to train for estimating bleed-through.
The available options are:

* `'Lasso'`: This was used for the paper linked above and is the default.
* `'ElasticNet'`: A generalization of `'Lasso'`.
* `'PoissonGLM'`: A `GLM` with the `PoissonRegressor` because pixel intensities follow a Poisson distribution.

For each channel, we train a separate instance of this model (on the selected pixels) using adjacent channels to estimate bleed-through.

### --channelOrdering

By default, we assumed that the order of channel numbers is the same as the order, in increasing wavelength, of the emission filters for those channels.
If this is not the case, use this parameter to specify, as a string of comma-separated integers, the wavelength-order of the channels.

For example, if the channels are `0, 1, 2, 3, 4` correspond to wavelengths (of the emission filter) of `420, 350, 600, 510, 580`, then the `--channelOrdering` should be `"1,0,3,4,2"`.

### --channelOverlap

For each channel in the image, we assume that the only noticeable bleed-through is from a small number of adjacent channels.
By default, we consider only `1` adjacent channel on each side of the wavelength scale as contributors to bleed-through.

For example, for channel 3, we would consider channels 2 and 4 to contribute bleed-through components.

Use a higher value for `--channelOverlap` to have our model look for bleed-through from farther channels.
This will, however, cause the models to take longer to train.

### --kernelSize

We learn a convolutional kernel for estimating the bleed-through from each channel to each neighboring channel.
This parameter specifies the size of those kernels.
It must be one of `"1x1"`, `"3x3"` or `"5x5"`.

Note that the time to train models scales with `kernelSize`.
Training with a "3x3" kernel takes 9 times as long as with a "1x1" kernel but also gives the best results.
Training with a "5x5" kernel takes 25 times as long as with a "1x1" kernel but, during testing, this produced 3x3 kernels with zero-padding around it.
We recommend "3x3" as the default.

## TODOs:

**1.**
There are some performance improvements to be made in how we select tiles and pixels to be trained.
We load tiles and sort pixels three times.
This can be cut down to loading tiles twice and sorting pixels only once.

**2.**
Since we are learning convolutional kernels here, we can re-frame the problem to use AutoEncoders or CNNs.
If successful, we can use deep-learning frameworks to greatly speed up the training process.

**3.**
ICA is an excellent candidate for this task.
However, I know of no way to train ICA with batches of data (with each set of tiles/pixels making a single batch).
We could probably run ICA if we had a small enough number of tiles/pixels but the memory-scaling of ICA generally makes it unsuitable for big-data applications.

**4.**
Test and add more options for selection criteria.

**5.**
The interaction components for two channels seem to be related.
Explore this relationship and possibly use it to link the models being trained.
For now, we train a model for each channel, and those models do not interact with each other.

**6.**
Create some synthetic datasets for testing the performance and breaking point of the algorithm.

**7.**
Add optional boolean parameter to look at one extra channel to the left, due to emission spectra having fat tails to the right.

**8.**
Add link to the paper.

**9.**
More testing around proper grouping of images by positional variables.

## Build the plugin

To build the Docker image for the conversion plugin, run `./build-docker.sh`.

## Install WIPP Plugin

In WIPP, navigate to the plugins page and add a new plugin.
Paste the contents of `plugin.json` into the pop-up window and submit.

## Options

This plugin takes 3 input arguments and 1 output argument:

| Name                   | Description                                          | I/O    | Type    | Default             |
|------------------------|------------------------------------------------------|--------|---------|---------------------|
| `--inpDir`             | Path to input images.                                | Input  | String  | N/A                 |
| `--filePattern`        | File pattern to subset images.                       | Input  | String  | N/A                 |
| `--groupBy`            | Variables to group together.                         | Input  | String  | N/A                 |
| `--selectionCriterion` | Method to use for selecting tiles.                   | Input  | Enum    | "HighMeanIntensity" |
| `--model`              | Model to train for estimating bleed-through.         | Input  | Enum    | "Lasso"             |
| `--channelOverlap`     | Number of adjacent channels to consider.             | Input  | Integer | 1                   |
| `--kernelSize`         | Size of convolutional kernels to learn.              | Input  | Enum    | "3x3"               |
| `--channelOrdering`    | Channel ordering by wavelength scale.                | Input  | String  | ""                  |
| `--outDir`             | Output image collection.                             | Output | String  | N/A                 |
| `--csvDir`             | Location for storing the bleed-through coefficients. | Output | String  | N/A                 |
