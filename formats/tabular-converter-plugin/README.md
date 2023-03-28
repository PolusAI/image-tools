# Tabular Converter (v0.1.0)

This WIPP plugin allows the tabular data conversion to `arrow` file format and vice versa. Currently this plugins handles only the vaex supported file formats.
This plugin supports the following file formats which are convertable into `arrow` file format:

1. fcs
2. csv
3. hdf5
4. fits
5. parquet
6. feather

However  the `arrow` file format is convertable to all other file formats except `fcs` and `fits`.
The support for additional file formats will be added in future.


Contact [Kelechi Nina Mezu](mailto:nina.mezu@nih.gov), [Hamdah Shafqat Abbasi](mailto:hamdahshafqat.abbasi@nih.gov) for more information.

For more information on WIPP, visit the [official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).

## Building

To build the Docker image for the conversion plugin, run
`bash build-docker.sh`.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin. Paste the
contents of `plugin.json` into the pop-up window and submit.

## Options

This plugin takes two input arguments and one output argument:

| Name          | Description             | I/O    | Type   |
|---------------|-------------------------|--------|--------|
| `--inpDir` | Input generic data collection to be processed by this plugin | Input | genericData |
| `--filePattern` | Pattern to parse tabular files | Input | string |
| `--fileExtension` | Desired pattern to convert | Input | string |
| `--outDir` | Output collection | Output | genericData |
| `--preview` | Generate JSON file with outputs | Output | JSON |
