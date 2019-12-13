# Montage

This plugin generates a stitching vector that will montage images together. The inputs are an image collection, a file pattern, and a layout array that indicates how the montage should be laid out. Below is a description of how the file pattern should be formatted and how to set the layout parameters.

For more information on WIPP, visit the [official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).

## Input Filename Pattern

This plugin uses [filename patterns](https://github.com/USNISTGOV/MIST/wiki/User-Guide#input-parameters) similar to that of what MIST uses. In particular, defining a filename variable is surrounded by `{}`, and the variable name and number of spaces dedicated to the variable are denoted by repeated characters for the variable. For example, if all filenames follow the structure `filename_TTT.ome.tif`, where TTT is a 0 padded number indicating the timepoint the image was captured at, then the filename pattern would be `filename_{ttt}.ome.tif`.

Filename patterns may include `x` and `y` grid positions for each image or a sequential position `p`, but not both. This differs from MIST in that `r` and `c` are used to indicate grid row and column rather than `y` and `x` respectively. This change was made to remain consistent with Bioformats dimension names and to permit use of `c` as a channel variable.

In addition to the position variables (both `x` and `y`, or `p`), the only other variables that can be used are `z`, `c`, `t`, and `r`.

## Montage Layout

The montage layout is specified by an array of strings. Each element in the array specifies the variables from the filename pattern that are used in every subgrid, starting from the most granular subgrid to the largest grid. This means that if the filename pattern is `filename_{xxx}_{yyy}_{ttt}.ome.tif` and the input array is `x,y,t`, this means that the images with the same `y` and `t` values but different `x` values are placed next to each other in a grid of images. Then, each grid of images is placed in another grid `t` is constant but `y` changes. Finally, all subgrids are placed into a grid according to `t`.

Each element in the layout array must contain one or two variables from a filename pattern. For example, if the filename pattern is `filename_{xxx}_{yyy}_{ttt}.ome.tif`, then a layout array could be `xy,t`. When variables are groups together, images are placed in the grid according to the variable values with one variable assigning the x-position and the other assigning the y-position. In the case of input layout `xy,t`, the smallest grid will place images with the same `t` value into a grid where the `x` variable indicating the x-position in the grid and the `y` variabled indicating the y-position. If the order of the variables was flipped (`yx`), then the positions of images would be transposed. There are no restrictions on which variables can be placed together, so it would be possible to have `xt,y` as a layout array.

## To do

**User specified grid spacing.** Currently, images in the smallest subgrid are placed 10 pixels apart, and each larger subgrid has spacing 4 times that of the previous subgrid. So, the first subgrid has spacing of 10 pixels, the 2nd has 40, the 4th will have 160.

**User defined grid shape.** Currently, grid dimensions are determined by the size of each dimension if two variable are present or the closest square that will fit all subgrids if one variable is present. So, if there are 48 images to be organized in a subgrid with a single variable, then images will be placed in a 7x7 grid.

## Building

To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin. Paste the contents of `plugin.json` into the pop-up window and submit.

## Options

This plugin takes one input argument and one output argument:

| Name          | Description             | I/O    | Type   |
|---------------|-------------------------|--------|--------|
| `--filePattern` | Filename pattern used to parse data | Input | string |
| `--inpDir` | Input image collection to be processed by this plugin | Input | collection |
| `--layout` | Specify montage organization | Input | array |
| `--outDir` | Output collection | Output | stitchingVector |
