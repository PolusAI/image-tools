# File Renaming(0.1.18-dev1)
This WIPP plugin uses supplied file naming patterns to dynamically
rename and save files in an image collection to a new image collection.

## Example Usage
* The user can upload an image collection where all files contain similar
naming conventions.

 * **Input collection:**
`img_x01_y01_DAPI.tif`
`img_x01_y01_GFP.tif`
`img_x01_y01_TXRED.tif`

 * **Output collection:**
`newdata_x001_y001_c001.tif`
`newdata_x001_y001_c002.tif`
`newdata_x001_y001_c003.tif`

 * **User input pattern:**
`img_x{row:dd}_y{col:dd}_{channel:c+}.ome.tif`

 * **User output pattern:**
`newdata_x{row:ddd}_y{col:ddd}_c{channel:ddd}.ome.tif`

* The user can format the output digit using the number of digits
specified in the output format.
 * `d` represents *digit*
 * `c` represents *character*.

* Note that c+ only matches letters in the alphabet, not symbols and numbers

* If the output formats have plus signs (+), then the number of output
digits/characters is not fixed.

* Finally, the input and output pattern data types *must* agree with one
exception:
 * If the input is a chracter and the output is digit,
then the script sorts the strings that match the character pattern and
assigns numbers 0+ to them.

* New optional feature `mapDirectory` implemented to include directory name in renamed files. Orignal directory name is added  in renamed files if `raw` value passed, `map` for mapped subdirectories `d0, d1, d2, ... dn` and `` for not including directory name in renamed files.


Contact [Melanie Parham](mailto:melanie.parham@axleinfo.com), [Hamdah Shafqat abbasi](mailto:hamdahshafqat.abbasi@nih.gov) for more
information.

For more information on WIPP, visit the
[official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).

## Building

To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin.
Paste the contents of `plugin.json` into the pop-up window and submit.

## Options

This plugin takes three input argument and one output argument:

| Name               | Description                       | I/O      | Type       |
|--------------------|-----------------------------------|----------|------------|
| `--inpDir`         | Input image collection            | Input    | collection |
| `--filePattern`    | Input filename pattern            | Input    | string     |
| `--outDir`         | Output collection                 | Output   | collection |
| `--outFilePattern` | Output filename pattern           | Input    | string     |
| `--mapDirectory`   | Directory name (`raw`, `map`, ``) | Input    | enum       |
| `--preview`        | Generate a JSON file with outputs | Output   | JSON       |
