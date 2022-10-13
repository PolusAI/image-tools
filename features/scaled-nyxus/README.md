# Scaled Nyxus


Scaled Nyxus plugin use parallel processing to extract nyxus features from intensity-label image data. Especially useful when processing high throughput screens.

Contact [Hamdah Shafqat Abbasi](mailto: hamdah.abbasi@axleinfo.com) for more information.

For more information on WIPP, visit the [official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).


## Building

To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin. Paste the
contents of `plugin.json` into the pop-up window and submit.

## Options

This plugin takes six input arguments and one output argument:

| Name               | Description                                                   | I/O    | Type          |
|--------------------|---------------------------------------------------------------|--------|---------------|
| `--inpDir`         | Input image directory                                         | Input  | collection    |
| `--segDir`         | Input label image directory                                   | Input  | collection    |
| `--filePattern`    | Filepattern to parse image replicates                         | Input  | string        |
| `--features`       | [nyxus features]('https://pypi.org/project/nyxus/')           | Input  | string        |
| `--neighborDist`   | Distance between two neighbor objects                         | Input  | float         |
| `--pixelPerMicron` | Pixel Size in micrometer                                      | Input  | float         |
| `--outDir`         | Output collection                                             | Output | collection    |




