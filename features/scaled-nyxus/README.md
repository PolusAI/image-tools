# Scaled Nyxus


Scaled Nyxus plugin uses parallel processing of [Nyxus python package](https://pypi.org/project/nyxus/) to extract nyxus features from intensity-label image data. Especially useful when processing high throughput screens.

Contact [Hamdah Shafqat Abbasi](mailto: hamdah.abbasi@axleinfo.com) for more information.

For more information on WIPP, visit the [official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).


# Note
Currently filepattern Python package is not implemented yet in the plugin and will be added later. Use a simplified regular expression to extract image replicates. There are five replicate images in the following below example \
*1-* `p001_x01_y01_wx01_wy01_c01.ome.tif`\
*2-* `p002_x01_y01_wx01_wy01_c01.ome.tif`\
*3-* `p003_x01_y01_wx01_wy01_c01.ome.tif`\
*4-* `p004_x01_y01_wx01_wy01_c01.ome.tif`\
*5-* `p005_x01_y01_wx01_wy01_c01.ome.tif`

Use `filePattern=p{p+}.*.ome.tif`


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
| `--features`       | [nyxus features](https://pypi.org/project/nyxus/)             | Input  | string        |
| `--neighborDist`   | Distance between two neighbor objects                         | Input  | float         |
| `--pixelPerMicron` | Pixel Size in micrometer                                      | Input  | float         |
| `--outDir`         | Output collection                                             | Output | collection    |