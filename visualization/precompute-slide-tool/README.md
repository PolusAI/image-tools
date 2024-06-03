# Polus Precompute Slide Plugin (1.7.2-dev0)

This WIPP plugin generates image pyramids in multiple viewing formats. Each
output has a special filepattern variable that will be used to combine images
for viewing in the output format.

1) **DeepZoom** - This file format creates time-slices of the data for viewing
in the Web Deep Zoom Toolkit. *Timepoints are designated by the `t` variable.*
2) **Neuroglancer** - This file format creates a 2D or 3D visualization for Neuroglancer. If images are already 3D, it will automatically create 3D image
stacks. *Depth layers are designated `z` variable.*
3) **Zarr** - This file format will create color visualizations for Polus
Render. *Channels are designated by the `c` variable.*

The file format can be specified in the filePattern input.
More details on the format: https://pypi.org/project/filepattern/

It assumes each image is a 2-dimensional plane, so it will not display an image
in 3D.

For more information on WIPP, visit the
[official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).

## Building

To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin. Paste the
contents of `plugin.json` into the pop-up window and submit.

## Options

This plugin can take four types of input argument and one output argument:

| Name          | Description                                           | I/O    | Type    |
| ------------- | ----------------------------------------------------- | ------ | ------- |
| `inpDir`      | Input image collection (Single Image Planes/Z Stacks) | Input  | Path    |
| `pyramidType` | DeepZoom/Neuroglancer/Zarr                            | Input  | Enum    |
| `filePattern` | Image pattern                                         | Input  | String  |
| `imageType`   | Neuroglancer type (Intensity/Segmentation)            | Input  | Enum    |
| `outDir`      | Output image pyramid                                  | Output | Pyramid |

### Pyramid Types

Must be one of the following:

- `DeepZoom`
- `Neuroglancer`
- `Zarr`

### Image Types

Must be one of the following:

- `Intensity`
- `Segmentation` (Neuroglancer only)
