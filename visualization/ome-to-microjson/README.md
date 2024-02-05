# Ome-to-microjson-plugin(v0.1.3-dev0)

This plugin generate a microjson of polygon coordinates of binary or label segmentations

This plugin uses [MICROJSON](https://github.com/bengtl/microjson/tree/dev) python library to generate polygon encodings for each object (Region of Interest) of binary segmented image in microjson format which can be used in [RENDER UI](https://render.ci.ncats.io/?imageUrl=https://files.scb-ncats.io/pyramids/Idr0033/precompute/41744/x(00-15)_y(01-24)_p0(1-9)_c(1-5)/)
application for visualization of microscopy images overlay with segmentations encodings.

Currently this plugin supports two Polygon types
1. rectangle
2. encoding
By default, this plugin uses `encoding` to compute polygon coordinates if `polygonType` is not defined.
`rectangle` polygon is bounding box coordinates, which surrounds each object and specifies its position.
`encoding` polygon is contour-based encoding, object boundries are represented as a series of connected line segements or curves.

## Examples

<img src="./image.png">

**a -** Example of Rectangle Polygon
**b -** Example of Encoding Polygon

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
| `inpDir`          | Input directory                                       | Input  | string         |
| `filePattern`     | Pattern to parse image filenames                    | Input  | string       |
| `polygonType`            | Polygon type (rectangle, encoding)                        | Input  | enum       |
| `outDir`          | Output directory                        | Output | string       |
| `preview`      | Generate a JSON file with outputs                     | Output | JSON            |

## Run the plugin

### Run the Docker Container

```bash
docker run -v /data:/data polusai/ome-to-microjson-plugin:0.1.3-dev0-dev \
  --inpDir /data/input \
  --filePattern ".*.ome.tif" \
  --polygonType "encoding" \
  --outDir /data/output \
  --preview
```
