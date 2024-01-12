# Nyxus-Plugin(v0.1.5-dev1)


Nyxus plugin uses parallel processing of [Nyxus python package](https://pypi.org/project/nyxus/) to extract nyxus features from intensity-label image data. Especially useful when processing high throughput screens.

Contact [Hamdah Shafqat Abbasi](mailto: hamdah.abbasi@axleinfo.com) for more information.

For more information on WIPP, visit the [official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).


## Note
Use two separate [filepatterns](https://filepattern.readthedocs.io/en/latest/) for intensity and label images.
For example if you have label images of one channel `c1`\
`segPattern='p00{z}_x{x+}_y{y+}_wx{t}_wy{p}_c1.ome.tif'`\
Use filepattern if you require to extract features from intensity images of all other channels\
`intPattern=p00{z}_x{x+}_y{y+}_wx{t}_wy{p}_c{c}.ome.tif`

## Output Format
Computed features outputs can be saved in either of formats `.csv`, `.arrow`, `.parquet` by passing values `pandas`, `arrowipc`, `parquet` to `fileExtension`. By default plugin saves outputs in `.csv`


## Building

To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin. Paste the
contents of `plugin.json` into the pop-up window and submit.

## Options

This plugin takes nine input arguments and one output argument:

| Name               | Description                                                        | I/O    | Type          |
|--------------------|--------------------------------------------------------------------|--------|---------------|
| `--inpDir`         | Input image directory                                              | Input  | collection    |
| `--segDir`         | Input label image directory                                        | Input  | collection    |
| `--intPattern`     | Filepattern to parse intensity images                              | Input  | string        |
| `--segPattern`     | Filepattern to parse label images                                  | Input  | string        |
| `--features`       | [nyxus features](https://pypi.org/project/nyxus/)                  | Input  | string        |
| `--fileExtension`  | A desired file format for nyxus features output                    | Input  | enum          |
| `--neighborDist`   | Distance between two neighbor objects                              | Input  | integer       |
| `--pixelPerMicron` | Pixel Size in micrometer                                           | Input  | float         |
| `--singleRoi`      | Treat intensity image as single roi and ignoring segmentation mask | Input  | bool          |
| `--outDir`         | Output collection                                                  | Output | collection    |
| `--preview`        | Generate a JSON file with outputs                                  | Output | JSON          |
