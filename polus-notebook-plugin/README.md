# Polus Jupyter Notebook Plugin

This plugin enables rapid code prototyping in Polus. Users create the code in Polus Notebooks in variety of languages and then run the notebook as a WIPP plugin.
Internally, Papermill is responsible for executing the notebook file.

For more information on WIPP, visit the [official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).

**This plugin is in development and is subject for change**

## Options

This plugin takes two input parameters and one output parameter:

| Name       | Description            | I/O    | Type |
|------------|------------------------|--------|------|
| `input`    | Input image collection | Input  | Path |
| `notebook` | Notebook filename      | Input  | Path |
| `output`   | Output image colee     | Output | List |


## Build the plugin

```bash
docker build . -t labshare/polus-notebook-plugin:0.1.2
```


## Run the plugin

### Manually

```bash
docker run -v /path/to/data:/data labshare/polus-notebook-plugin:0.1.2 \
  --input /data/input \
  --notebook /data/notebook.ipynb \
  --output /data/output
```