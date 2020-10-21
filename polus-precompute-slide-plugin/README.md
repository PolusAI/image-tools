# Polus Precompute Volume Plugin

This WIPP plugin turns all tiled tiff images in an image collection into a [Neuroglancer precomputed format](https://github.com/google/neuroglancer/tree/master/src/neuroglancer/datasource/precomputed). It assumes each image is a 2-dimensional plane, so it will not display an image in 3D. The tiled tiff format and associated metadata is accessed using Bioformats.

_**This plugin is not a major release version.**_ A breaking change may occur when Neuroglancer is implemented into a WIPP deployment or if the data type for Neuroglancer precomputed plugins is changed to something other than `pyramid`. Currently, the output from this plugin is a `pyramid`, but WIPP will attempt to open the output of this plugin in WDZT. Either a new data type will need to be created inside of WIPP, or an option to open up the pyramid using Neuroglancer will need to be implemented.

For more information on WIPP, visit the [official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).

For more information on Bioformats, vist the [official page](https://www.openmicroscopy.org/bio-formats/).

## To do

1. Additional parallelization: Currently the plugin splits the generation of each image pyramid off into its own process. It would be more memory efficient and faster to build individual pyramids if subpyramids were built in separate pyramids. This would require the creation of a method to read pyramid tiles and possibly a dag-like structure to check that certain tiles were created before starting a process.
2. GPU acceleration: Since the number of disk reads is kept at a minimum and the main computational load is averaging pixels together to build lower resolution images, this plugin is a good candidate for gpu-acceleration.

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

## Run the plugin

### Run the Docker Container

```bash
docker run -v /path/to/data:/data labshare/polus-precomputed-slide-plugin \
  --inpDir /data/input \
  --outDir /data/output
```
