# Tabular Converter (v0.1.0)

This WIPP plugin allows researches to convert the tabular data to various vaex supported file formats. 
This plugin supports the following file extensions are convertable into Arrow Feather File Format (V2):
- `fcs`
- `csv`
- `hdf5`
- `fits`
- `parquet`
- `feather`

However  Arrow Feather File Format (V2) is convertable to all other file extensions except `fcs` and `fits`. The support for other file extensions will be added in future.


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
| `--inpDir` | Input generic data collection to be processed by this plugin | Input | collection |
| `--filePattern` | Desired pattern to convert | Input | string |
| `--outDir` | Output collection | Output | collection |
| `--preview` | Generate JSON file with outputs | Output | JSON |
