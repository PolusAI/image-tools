# Polus Tiled Tiff Conversion Plugin

This WIPP plugin takes any image type supported by Bioformats and converts it to
an OME tiled tiff. For file formats that have a pyramid like structured with
multiple resolutions (or series), this plugin only saves the first series to a
tiff (usually the first series is the highest resolution).

The current need for this plugin is that WIPPs tiled tiff conversion process
only grabs the first image plane, while this plugin grabs all image planes in a
series. Ultimately, this permits complete conversion of data (including all
channels, z-positions, and time-points). Each image plane (defined as a single
z-slice, channel, or timepoint) is saved as a separate image. If an image has
multiple slices for a given dimension (z-slice, channel, or timepoint), then an
indicator of which slice is appended to the file name. For example, an image
named `Image.czi` that has two z-slices, two channels, and two timepoints will
have the following files exported by this plugin:

```bash
Image_z0_c0_t0.ome.tif
Image_z1_c0_t0.ome.tif
Image_z0_c1_t0.ome.tif
Image_z1_c1_t0.ome.tif
Image_z0_c0_t1.ome.tif
Image_z1_c0_t1.ome.tif
Image_z0_c1_t1.ome.tif
Image_z1_c1_t1.ome.tif
```

For more information on WIPP, visit the
[official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).

## Building

To build from source code, run `./mvn-packager.sh`

To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin. Paste the
contents of `plugin.json` into the pop-up window and submit.

## Options

This plugin takes one input argument and one output argument:

| Name          | Description                   | I/O    | Type    | Required |
| ------------- | ----------------------------- | ------ | ------- | -------- |
| `input`       | Input image collection        | Input  | Path    | true     |
| `output`      | Output image collection       | Output | Path    | true     |

## Run the plugin

### Run the Docker Container

```bash
docker run -v /path/to/data:/data labshare/polus-tiledtiff-converter-plugin:1.1.0 \
  --input /data/input \
  --output /data/output
```
