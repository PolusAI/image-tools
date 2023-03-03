# Polus Rencer Overlay Plugin

This WIPP plugin generates text overlay data which can be used in render. The
text overlays are generated for a given stitching vector(s). For each grid and 
all subsequent sub-grids text overlay data will be created. The smallest grid 
will always be a single image (single FOV). This plugin determines the grid 
layout, or Montage layout, which was establised during the image assembly. More 
information about Montage grid layout can be found at the following link,
starting on slide 22 (Montage Layout).

[Montage Layout](https://docs.google.com/presentation/d/1DJrtu8EQgm5V32OC_M4BfFvtMdr5AkVtK3dAJZIwybw/edit#slide=id.g805758cc76_0_39)

Note that if two variables define a grid (x and y for example) the text overlay
will follow a [bijective](https://en.wikipedia.org/wiki/Bijective_numeration) 
numeration format i.e. `A1`,`A2`,`A3`,...,`Z1`,`Z2`,`Z3`,...

For single varbiale grids the plugin will use the upper case version of the 
variable in the filepattern plus an integer (n) which represents the nth image
or grid in the parent grid; i.e. `C1`, `C2`, `C3`, `C4`,...

The file format can be specified in the filePattern input. This should be the
same file pattern that was used to assemble the images.
More details on the format: https://pypi.org/project/filepattern/

This plugin has an optional input which will combine all of the output `.json`
files into a single file named `overlay.json`. Either way a `.json` file will
be created for every stitching vector in the input directory. The name of each
file will follow the filepattern name with the range of the variables which is
generated from a the filepattern; i.e. `p01_x(01-24)_y(01-16)_c(1-4).json`.

For more information on WIPP, visit the
[official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).

## Building

To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin. Paste the
contents of `plugin.json` into the pop-up window and submit.

## Options

This plugin can take three types of input argument and one output argument:

| Name              | Description                                           | I/O    | Type         |
|-------------------|-------------------------------------------------------|--------|--------------|
| `stitchingVector` | Stitching vector(s) image collection                  | Input  | Path         |
| `filePattern`     | Image pattern                                         | Input  | String       |
| `concatenate`     | If all output files should be concatenated            | Input  | String       |
| `heatmap`         | If heatmap overlay data should be generated           | Input  | String       |
| `chem`            | If chemical overlay data should be generated          | Input  | String       |
| `text`            | If text overlay data should be generated              | Input  | String       |
| `outDir`          | Output directory for overlays                         | Output | String       |

## Run the plugin

### Run the Docker Container

```bash
docker run -v /path/to/data:/data polusai/polus-precomputed-slide-plugin \
  --inpDir /data/input \
  --outDir /data/output
```
