# Rencer Overlay Plugin(v0.1.0-dev1)

This plugin uses [MICROJSON](https://github.com/bengtl/microjson/tree/dev) python library to generate overlays in JSON format which can be used in
[RENDER UI](https://render.ci.ncats.io/?imageUrl=https://files.scb-ncats.io/pyramids/Idr0033/precompute/41744/x(00-15)_y(01-24)_p0(1-9)_c(1-5)/)
application for visualization of microscopy images.

This plugin allows to calculate geometry coordinates based on the values passes for input arguments i-e `type`, `cellWidth`, `cellHeight` for each row
 and column positions of user-defined microplate `dimensions`.
Note: Currently this plugin supports two geometry types `Polygon` and `Point`.A future work requires addtional support of more geometry types in this plugin.

Currently this plugins handles only three file formats supported by vaex.
1. csv
2. arrow
3. feather


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

This plugin can take six input arguments and one output argument:

| Name              | Description                                           | I/O    | Type         |
|-------------------|-------------------------------------------------------|--------|--------------|
| `inpDir`          | Input directory                                       | Input  | string         |
| `filePattern`     | Pattern to parse tabular filenames                    | Input  | string       |
| `dimensions`      | Select microplate type i-e (384, 96, 24, 6) well plate  | Input  | string       |
| `type`            | Geometry type (Polygon, Point)                        | Input  | string       |
| `cellWidth`       | Pixel distance between adjacent cells/wells in x-dimension | Input  | integer       |
| `cellHeight`       | Pixel distance in y-dimension              | Input  | integer       |
| `outDir`          | Output directory for overlays                         | Output | string       |
| `--preview`      | Generate a JSON file with outputs                     | Output | JSON            |

## Run the plugin

### Run the Docker Container

```bash
docker run -v /path/to/data:/data polusai/render-overlay-plugin:0.1.0-dev1 \
  --inpDir /data/input \
  --filePattern ".*" \
  --dimension 384 \
  --type "Polygon" \
  --cellWidth 2170 \
  --cellHeight 2180 \
  --outDir /data/output \
  --preview
```
