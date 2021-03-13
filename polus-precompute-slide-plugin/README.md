# Polus Precompute Slide Plugin

This WIPP plugin turns all tiled tiff images in an image collection into a
[Neuroglancer precomputed format](https://github.com/google/neuroglancer/tree/master/src/neuroglancer/datasource/precomputed).
It assumes each image is a 2-dimensional plane, so it will not display an image
in 3D. The tiled tiff format and associated metadata is accessed using
Bioformats.

For more information on WIPP, visit the
[official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).

## Building

To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin. Paste the
contents of `plugin.json` into the pop-up window and submit.

## Options

This plugin can take two types of input argument and one output argument:

| Name          | Description                                           | I/O    | Type    |
|---------------|-------------------------------------------------------|--------|---------|
| `inpDir`      | Input image collection (Single Image Planes/Z Stacks) | Input  | Path    |
| `pyramidType` | DeepZoom/Neuroglancer                                 | Input  | String  |
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
