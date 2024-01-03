# Image dimension stacking

This WIPP plugin uses a file name pattern to identify either z-slices in a 3D volume, in
an image collection and then create a new image collection with the z-slices
combined into a single volume (using Bioformats tiled tiff format).

This plugin uses regular expressions similar to what MIST uses for filenames,
but there are important exceptions that are described below in the section
titled **Input Filename Pattern**.

For more information on WIPP, visit the
[official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).

## Building

To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## Input Filename Pattern

This plugin uses the
[filepattern](https://github.com/LabShare/polus-plugins/tree/master/utils/polus-filepattern-util)
utility to indicate which files to stack. In particular, defining a filename
variable is surrounded by `{}`, and the variable name and number of spaces
dedicated to the variable are denoted by repeated characters for the variable.
For example, if all filenames follow the structure `filename_ZZZ.ome.tif`, where
ZZZ indicates the timepoint the image was captured at, then the filename pattern
would be `filename_{zzz}.ome.tif` or `filename_{z+}.ome.tif`, where former
indicates the z-position is always of uniform length (e.g.
`filename_001.ome.tif`) while the latter permits variable z-position length.

The only required variable for this plugin is `z`. Filename patterns may include
any of the other filepattern variables, mixed with any other valid regular
expression arguments (except groups).


## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin. Paste the
contents of `plugin.json` into the pop-up window and submit.

## Options

This plugin takes three input argument and one output argument:

| Name          | Description             | I/O    | Type   |
|---------------|-------------------------|--------|--------|
| `--inpDir`      | Input image collection  | Input  | Collection   |
| `--filePattern` | Pattern to parse image files           | Input  | String |
| `--groupBy` | A variable to group image files           | Input  | String |
| `--outDir`      | Output image collection | Output | Collection   |
| `--preview`        | Generate a JSON file to view outputs | Output | Boolean   |

### Run the Docker Container

```bash
docker run -v /path/to/data:/data polusai/image-dimension-stacking-plugin:0.1.0-dev \
  --inpDir "Path/To/Data" \
  --filePattern "tubhiswt_C1-z{z:d+}.ome.tif" \
  --groupBy "c" \
  --outDir "Path/To/Output/Dir"
```
