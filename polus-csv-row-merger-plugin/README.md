# CSV Row Merger

This WIPP plugin merges all csv files in a csv collection into a single csv file, merging along rows. If some csv files have columns that do not match those of others, the missing columns are filled with 'NaN' values. A column titled `file` is created in the output file, and this contains the name of the original input csv file associated with the row of data. If `stripExtension` is set to true, then the `.csv` file extension is removed from the file name in the `file` column.

For more information on WIPP, visit the [official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).

## Building

To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin. Paste the contents of `plugin.json` into the pop-up window and submit.

## Options

This plugin takes two input argument and one output argument:

| Name          | Description             | I/O    | Type   |
|---------------|-------------------------|--------|--------|
| `--inpDir` | Input image collection to be processed by this plugin | Input | collection |
| `--stripExtension` | Should csv be removed from the filename in the output file | Input | boolean |
| `--outDir` | Output csv file | Output | csvCollection |

