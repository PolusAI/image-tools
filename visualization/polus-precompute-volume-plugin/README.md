# Polus Precompute Volume Plugin

This WIPP plugin turns all tiled tiff images in an image collection into a [Neuroglancer precomputed format](https://github.com/google/neuroglancer/tree/master/src/neuroglancer/datasource/precomputed). The tiled tiff format and associated metadata is accessed using bfio and uses the third party library neurogen. 

This plugin can also creates meshes if the imagetype is 'segmentation'

For more information on Bioformats, vist the [official page](https://www.openmicroscopy.org/bio-formats/).

## Building

To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin. Paste the contents of `plugin.json` into the pop-up window and submit.

## Options

This plugin takes one input argument and one output argument:

| Name       | Description             | I/O    | Type |
|------------|-------------------------|--------|------|
| `inpDir`   | Input image collection  | Input  | Path |
| `outDir`   | Output image pyramid    | Output | Pyramid |
| `imageType`   | Image/Segmentation  | Input  | String |
| `mesh` | Generate Mesh for Labelled Data | Input | Boolean |
| `imagepattern`   | Image pattern  | Input  | String |

