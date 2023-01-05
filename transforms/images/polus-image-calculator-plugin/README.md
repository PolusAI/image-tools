# Image Calculator (v0.1.0)

This plugin performs pixelwise operations between two image collections. For
example, images in one image collection are subtracted from images in another
collection.

For more information on WIPP, visit the
[official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).

## TODO

1. Enable simple matching rather than relying only on variables in filepatterns
2. Size/type checking of both images the operation is applied to. Currently, the
plugin assumes the images are the same size and data type. Also, data type
overflow is not currently handled.

## Building

To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin. Paste the
contents of `plugin.json` into the pop-up window and submit.

## Options

This plugin takes 5 input arguments and
1 output argument:

| Name          | Description             | I/O    | Type   |
|---------------|-------------------------|--------|--------|
| `--primaryDir` | The first set of images | Input | collection |
| `--primaryPattern` | Filename pattern used to separate data | Input | string |
| `--operator` | The operation to perform | Input | enum |
| `--secondaryDir` | The second set of images | Input | collection |
| `--secondaryPattern` | Filename pattern used to separate data | Input | string |
| `--outDir` | Output collection | Output | collection |

