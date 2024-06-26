# Remove border objects


Remove border objects plugin clear objects which touch image borders and squentially relabelling of image objects

Contact [Hamdah Shafqat Abbasi](mailto: hamdah.abbasi@axleinfo.com) for more information.

For more information on WIPP, visit the [official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).

## To Do
At the moment this plugin supports label images with two dimensions only. We will add support for higher dimensions later.


## Description

<img src="./image.png">


**a -** Original image contains 67 unique label objects 
**b -** Image with 16 detected border objects
**c -** Removing Border objects and sequential relabelling


## Building

To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin. Paste the
contents of `plugin.json` into the pop-up window and submit.

## Options

This plugin takes two input arguments and
1 output argument:

| Name          | Description                                                   | I/O    | Type          |
|---------------|---------------------------------------------------------------|--------|---------------|
| `--inpDir`    | Input image directory                                         | Input  | collection    |
| `--pattern`   | Filepattern to parse image files                              | Input  | string        |
| `--outDir`    | Output collection                                             | Output | collection    |




