# Tabular to Feather (v0.1.1)
This WIPP plugin allows analysts to convert various file formats received by researchers into Arrow Feather File Format (V2). This plugin supports the following file extensions: 
    - `.fcs`
    - `.csv`
    - `.hdf5`
    - `.fits`
    - `.parquet`
    - `.arrow`

Contact [Kelechi Nina Mezu](mailto:nina.mezu@nih.gov) for more information.

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
| `--filePattern` | Filename pattern to convert | Input | string |
| `--inpDir` | Input generic data collection to be processed by this plugin | Input | collection |
| `--outDir` | Output collection | Output | collection |