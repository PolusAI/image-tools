# Image dimension stacking(0.1.0-dev)

This plugin leverages the [filepattern](https://filepattern2.readthedocs.io/en/latest/Home.html) library and employs the `groupBy` variable to enable the matching of image filenames, facilitating their stacking into multi-dimensional images.
Currently, the plugin supports the following dimensions and user can choose the relevant variable for the `groupBy` input argument.
1. multi-channel  `groupBy=c`\
   For example `filePattern=x01_y01_p01_c{c:d+}.ome.tif`
2. multi-zplanes  `groupBy=z`\
   For example `filePattern=tubhiswt_C1-z{z:d+}.ome.tif`
3. multi-timepoints  `groupBy=t`\
   For example `filePattern=img00001_t{t:d+}_ch0.ome.tif`

#### Note:
Filename patterns may consist of any other filepattern variables, combined with other valid regular expression arguments, excluding the `groupBy` variable.

For more information on WIPP, visit the
[official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).

## Building

To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin. Paste the
contents of `plugin.json` into the pop-up window and submit.

## Options

This plugin takes three input argument and one output argument:

| Name          | Description             | I/O    | Type   |
|---------------|-------------------------|--------|--------|
| `--inpDir`      | Input image collection  | Input  | Collection   |
| `--filePattern` | Pattern to parse image files           | Input  | String |
| `--groupBy` | A variable to group image files           | Input  | String |
| `--outDir`      | Output image collection | Output | Collection   |
| `--preview`        | Generate a JSON file to view outputs | Output | Boolean   |

### Run the Docker Container

```bash
docker run -v /path/to/data:/data polusai/image-dimension-stacking-plugin:0.1.0-dev \
  --inpDir "Path/To/Data" \
  --filePattern "tubhiswt_C1-z{z:d+}.ome.tif" \
  --groupBy "z" \
  --outDir "Path/To/Output/Dir"
```
