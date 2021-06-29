# Polus Precompute Slide Plugin

This WIPP can generate pyramids for three different types of data:

1) DeepZoom
*    This file format creates time-slices of the data. (Stacks by the 't' dimension)
2) Neuroglancer 
*    This file format creates a 3D volume of the data. (Stacks by the 'z' dimension)
3) Zarr
*    This file format stacks the images by its channel. (Stacks by the 'c' dimension)


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
|---------------|-------------------------------------------------------|--------|---------|
| `inpDir`      | Input image collection (Single Image Planes/Z Stacks) | Input  | Path    |
| `pyramidType` | DeepZoom/Neuroglancer/Zarr                                 | Input  | String  |
| `filePattern` | Image pattern                                         | Input  | String  |
| `imageType`   | Neuroglancer type (image/segmentation)                | Input  | String  |
| `outDir`      | Output image pyramid                                  | Output | Pyramid |

## Run the plugin

### Run the Docker Container

```bash
docker run -v /path/to/data:/data labshare/polus-precomputed-slide-plugin \
  --inpDir /data/input \
  --outDir /data/output
```
