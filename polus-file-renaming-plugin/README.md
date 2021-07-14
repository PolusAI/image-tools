# File Renaming
This WIPP plugin uses user-supplied file naming patterns to dynamically rename and save files in an image collection to a new output directory. 

## Example Usage
The user can upload an image collection where all files contain similar naming conventions. 

**Input collection:**
`img_x01_y01_DAPI.tif`
`img_x01_y01_GFP.tif`
`img_x01_y01_TXRED.tif`

**Output collection:**
`newdata_x001_y001_c001.tif`
`newdata_x001_y001_c002.tif`
`newdata_x001_y001_c003.tif`

**User input pattern:**
`img_x{row:dd}_y{col:dd}_{channel:c+}.ome.tif`

**User output pattern:**
`newdata_x{row:ddd}_y{col:ddd}_c{channel:ddd}.ome.tif`

The user can format the output digit using the number of digits specified in the output format.
* `d` or `i` represent *digit*/*integer*
* `c` represents *character*
* `f` represents *floating point*

If the output formats have plus signs (+), then the number of output digits/characters is not fixed.

Finally, the input and output pattern data types *must* agree with one exception:
* If the input is a chracter and the output is digit, then the script sorts the strings that match the character pattern and assigns numbers 1+ to them.


Contact [Melanie Parham](mailto:melanie.parham@axleinfo.com) for more information.

For more information on WIPP, visit the [official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).

## Building

To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin. Paste the contents of `plugin.json` into the pop-up window and submit.

## Options

This plugin takes one input argument and one output argument:

| Name          | Description             | I/O    | Type   |
|---------------|-------------------------|--------|--------|
| `--inpDir` | Input image collection to be processed by this plugin | Input | collection |
| `--outDir` | Output collection | Output | collection |
| `--filePattern` | Filename pattern used to separate data | Input | string |
| `--outFilePattern` | Desired filename pattern used to rename and separate data | Input | string |