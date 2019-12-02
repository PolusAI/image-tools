# Stack Z-Slices

This WIPP plugin will uses a file name pattern to identify z-slices in a 3D volume in an image collection and then create a new image collection with the z-slices combined into a single volume (using Bioformats tiled tiff format).

This plugin uses regular expressions similar to what MIST uses for filenames, but there are important exceptions that are described below in the section titled **Input Filename Pattern**.

For more information on WIPP, visit the [official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).

## Building

To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## Input Filename Pattern

This plugin uses [filename patterns](https://github.com/USNISTGOV/MIST/wiki/User-Guide#input-parameters) similar to that of what MIST uses. In particular, defining a filename variable is surrounded by `{}`, and the variable name and number of spaces dedicated to the variable are denoted by repeated characters for the variable. For example, if all filenames follow the structure `filename_TTT.ome.tif`, where TTT indicates the timepoint the image was captured at, then the filename pattern would be `filename_{ttt}.ome.tif`.

The only required variable for this plugin is `z`. Filename patterns may include `x` and `y` grid positions for each image or a sequential position `p`, but not both. This differs from MIST in that `r` and `c` are used to indicate grid row and column rather than `y` and `x` respectively. This change was made to remain consistent with Bioformats dimension names and to permit use of `c` as a channel variable.

In addition to the position variables (both `x` and `y`, or `p`), the only other variables that can be used are `z`, `c`, and `t`. Images with the same `x`, `y`, `t`, and `c` values will be compiled into a single tiled tif.

## To do

This plugin does not set the physical z-dimension value, which would indicate the distance between z-slices. If the original files included a z-dimension, then this value will be used. If not, then the z-value will default to 1. It would be useful to have an additional input that would set the physical z distance between slices.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin. Paste the contents of `plugin.json` into the pop-up window and submit.

## Options

This plugin takes one input argument and one output argument:

| Name          | Description             | I/O    | Type   |
|---------------|-------------------------|--------|--------|
| `inpDir`      | Input image collection  | Input  | Path   |
| `filePattern` | File pattern            | Input  | String |
| `outDir`      | Output image collection | Output | List   |

### Run the Docker Container

```bash
docker run -v /path/to/data:/data labshare/stack-z-slices \
  --inpDir "Path/To/Data" \
  --filePattern "File_x{xxx}_y{yyy}_z{zzz}.ome.tif" \
  --outDir "Path/To/Output/Dir"
```
