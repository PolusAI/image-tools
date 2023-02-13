# Polus CZI Extraction Plugin

This WIPP plugin will import a csv collection and build a DeepZoom pyramid of graphs, where each graph contains a heatmap of each column plotted against another column. All n-columns are plotted against each other, excluding tranposed graphs and graphs where each axis has the same column. This leads to a total of (n^2-n)/2 graphs.

Two types of graphs will be produced: 
1) Linear sclaed graphs
2) Log scaled graphs

  The output will contain dzi and csv files for both linear and log scaled outputs. 
  There were will be two different directories that contain the pyramid images for the linear and log scaled outputs

For more information on WIPP, visit the [official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).

## Building

To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin. Paste the contents of `plugin.json` into the pop-up window and submit.

## Options

This plugin takes one input argument and one output argument:

| Name     | Description            | I/O    | Type             |
| -------- | ---------------------- | ------ | ---------------- |
| `inpDir` | Input CSV   collection | Input  | CSV   Collection |
| `outDir` | Output pyramid         | Output | Pyramid          |

## Run the plugin

### Run the Docker Container

```bash
docker run -v /path/to/data:/data graph-pyramid-builder \
  --inpDir /data/input \
  --outDir /data/output
```
