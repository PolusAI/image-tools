# Polus Jupyter Notebook Plugin

This plugin enables rapid code prototyping in Polus. Users create the code in Polus Notebooks in variety of languages and then run the notebook as a WIPP plugin.
Internally, Papermill is responsible for executing the notebook file.

For more information on WIPP, visit the [official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).

**This plugin is in development and is subject for change**

## Options

This plugin takes two input parameters and one output parameter:

| Name       | Description             | I/O    | Type |
|------------|-------------------------|--------|------|
| `input`    | Input image collection  | Input  | Path |
| `notebook` | Notebook filename       | Input  | Path |
| `output`   | Output image collection | Output | List |


## Build the plugin

```bash
docker build . -t labshare/polus-notebook-plugin:0.1.3
```


## Run the plugin

### Manually

```bash
docker run -v /path/to/data:/data labshare/polus-notebook-plugin:0.1.3 \
  --input /data/input \
  --notebook /data/notebook.ipynb \
  --output /data/output
```

## Known issues

- Papermill only supports Python, R and Matlab notebooks
- Custom translators are added to support Polyglot SoS Notebooks, however Notebook parameters (input and output paths) can only be set in SoS or Python cells of Polyglot notebook.
- Bash cells in SoS notebooks are failing and stopping the execution of all the cells below the Notebook without throwing a global error (Workflow is marked as succeeded)