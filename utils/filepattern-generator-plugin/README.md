# Filepattern Generator


Filepattern Generator plugin creates a json containing a number of new filepatterns, where each filepattern will subset the image data in the directory

Contact [Nick Schaub , Hamdah Shafqat Abbasi](mailto:nick.schaub@nih.gov, hamdah.abbasi@axleinfo.com) for more information.

For more information on WIPP, visit the [official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).


## Building

To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin. Paste the
contents of `plugin.json` into the pop-up window and submit.

## Options

This plugin takes 5 input arguments and
1 output argument:

| Name          | Description                                                   | I/O    | Type          |
|---------------|---------------------------------------------------------------|--------|---------------|
| `--inpDir`    | Input image directory                                         | Input  | collection    |
| `--pattern`   | Filepattern to parse image files                              | Input  | string        |
| `--chunkSize` | Number of images to generate collective filepattern           | Input  | number        |
| `--groupBy`   | Select a parameter to generate filepatterns in specific order | Input  | string        |
| `--outDir`    | Output generic collection                                     | Output | genericData    |




