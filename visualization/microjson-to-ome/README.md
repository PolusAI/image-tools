# Microjson To Ome(v0.1.3-dev0)

This plugin reconstructs binary image from polygon coordinates reserved in microjson file format

Currently this plugin supports binary image reconstruction for only two Polygon types
1. rectangle
2. encoding
`rectangle` polygon is bounding box coordinates, which surrounds each object and specifies its position.
`encoding` polygon is contour-based encoding, object boundries are represented as a series of connected line segements or curves.

## Examples

<img src="./image.png">

**a -** Microjson file with poly coordinates of objects
**b -** Reconstructed image from polygon coordinates

Contact [Hamdah Shafqat Abbasi](mailto:hamdahshafqat.abbasi@nih.gov) for more information.
For more information on WIPP, visit the
[official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).

## Building

To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin. Paste the
contents of `plugin.json` into the pop-up window and submit.

## Options

This plugin can take four input arguments and one output argument:

| Name              | Description                                           | I/O    | Type         |
|-------------------|-------------------------------------------------------|--------|--------------|
| `inpDir`          | Input directory to                                       | Input  | genericData         |
| `filePattern`     | Pattern to parse image filenames                    | Input  | string       |
| `outDir`          | Output directory                        | Output | collection       |
| `--preview`      | Generate a JSON file with outputs                     | Output | JSON            |

## Run the plugin

### Run the Docker Container

```bash
docker run -v /data:/data polusai/microjson-to-ome-plugin:0.1.2-dev \
  --inpDir /data/input \
  --filePattern ".*.json" \
  --outDir /data/output \
  --preview
```
