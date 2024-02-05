# Cell Border Segmentation(v0.2.2-dev0)

This plugin segments epithelial cell borders which are labeled with fluorescent proteins that highlight cell borders while leaving the cell body dark.
This was orginally designed to segment retinal epithelial cells, fluorescently labeled with zonula occluden-1 (ZO1) tight junction protein. The magnifications this was trained on varied from 10x to 40x, so it should work well on a wide range of magnifications.
epithelial

The segmentation algorithm is a neural network, and it was trained on cells retinal pigment epithelial cells from multiple organisms, from multiple labs, different microscopes, and at multiple magnifications. The neural network used in this plugin was originally reported in the publication ["Deep learning predicts function of live retinal pigment epithelium from quantitative microscopy"](https://www.jci.org/articles/view/131187).

The data used to train the neural network is freely available [here](https://doi.org/doi:10.18434/T4/1503229).

For more information on WIPP, visit the [official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).

## Building

To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin. Paste the contents of `plugin.json` into the pop-up window and submit.

## Options

This plugin takes two input argument and one output argument:

| Name       | Description                                           | I/O    | Type       |
| ---------- | ----------------------------------------------------- | ------ | ---------- |
| `--inpDir` | Input image collection to be processed by this plugin | Input  | collection |
| `--filePattern` | Pattern to parse image files | Input  | string |
| `--outDir` | Output collection                                     | Output | collection |
## Examples
<img src="./img.png">

### To Download Z01 Flurorescent test dataset from Publication
```Linux
wget "https://isg.nist.gov/deepzoomweb/dissemination/rpecells/fluorescentZ01.zip"
unzip fluorescentZ01.zip
```

### Run the Docker Container

```bash
docker run -v /data:/data polusai/zo1-border-segmentation-plugin:0.2.2-dev0 \
  --inpDir /data/input \
  --filePattern ".*.ome.tif" \
  --outDir /data/output \
  --preview
```
