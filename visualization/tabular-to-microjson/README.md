# Tabular To Microjson(v0.1.1)

This plugin uses [MICROJSON](https://github.com/bengtl/microjson/tree/dev) python library to generate JSON from tabular data which can be used in
[RENDER UI](https://render.ci.ncats.io/?imageUrl=https://files.scb-ncats.io/pyramids/Idr0033/precompute/41744/x(00-15)_y(01-24)_p0(1-9)_c(1-5)/)
application for visualization of microscopy images.

This plugin allows to calculate geometry coordinates i-e `Polygon` and `Point` using image positions from corresponding stitching vector.
Note: The filenames of tabular and stitching vector should be same
`groupBy` is used when there are more than one image in each well then pass a `variable` used in `stitchPattern` to group filenames in a stitching vector to compute geometry coordinates.

Note: Currently this plugin supports two geometry types `Polygon` and `Point`.A future work requires additional support of more geometry types in this plugin.

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

This plugin can take seven input arguments and one output argument:

| Name              | Description                                           | I/O    | Type         |
|-------------------|-------------------------------------------------------|--------|--------------|
| `inpDir`          | Input directory                                       | Input  | string         |
| `stitchDir`       | Directory containing stitching vectors                | Input  | string         |
| `filePattern`     | Pattern to parse tabular filenames                    | Input  | string       |
| `stitchPattern`   | Pattern to parse filenames in stitching vector        | Input  | string       |
| `groupBy`         | Variable to group filenames in  stitching vector | Input  | string       |
| `geometryType`    | Geometry type (Polygon, Point)                        | Input  | string       |
| `outDir`          | Output directory for overlays                         | Output | string       |
| `--preview`      | Generate a JSON file with outputs                     | Output | JSON            |

## Run the plugin

### Run the Docker Container

```bash
docker run -v /data:/data polusai/tabular-to-microjson-plugin:0.1.1 \
  --inpDir /data/input \
  --stitchDir /data/stitchvector \
  --filePattern ".*.csv" \
  --stitchPattern "x{x:dd}_y{y:dd}_c{c:d}.ome.tif" \
  --groupBy None \
  --geometryType "Polygon" \
  --outDir /data/output \
  --preview
```
