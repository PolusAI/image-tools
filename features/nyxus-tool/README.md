# Nyxus-tool(v0.1.9-dev0)


Parallelized feature extraction from intensity + label image pairs using the **[Nyxus](https://pypi.org/project/nyxus/)** library.

Especially useful for high-throughput microscopy screens.

Contact [Hamdah Shafqat Abbasi](mailto: hamdah.abbasi@axleinfo.com) for more information.

For more information on WIPP, visit the [official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).


## Important notes

- Use two separate **[filepattern](https://filepattern.readthedocs.io/en/latest/)** for intensity and label images.
- Example naming scheme:

    Intensity (multi-channel):
    `intPattern=p00{z}_x{x+}_y{y+}_wx{t}_wy{p}_c{c}.ome.tif`

    Segmentation :
    `segPattern='p00{z}_x{x+}_y{y+}_wx{t}_wy{p}_c1.ome.tif'`

- `--singleRoi` mode treats each intensity image as one whole-object ROI (ignores segmentation mask)
- Nyxus parameters (e.g., `neighbor_distance`, `pixels_per_micron`) are passed via repeatable `--kwargs KEY=VALUE`
- Output file extension (format) is controlled via environment variable `POLUS_TAB_EXT` (default: `.csv`; options: `.csv`, `.arrow`, `.parquet`)


## Building

To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin. Paste the
contents of `plugin.json` into the pop-up window and submit.

## Quick run example (Docker)

See `run-plugin.sh` for a template.


```bash
docker run --rm -v /path/to/data:/data \
  -e POLUS_TAB_EXT=pandas \
  polusai/nyxus-tool:0.1.9-dev0 \
  --inpDir      /data/intensity \
  --segDir      /data/segmentation \
  --intPattern  'p00{z}_x{x+}_y{y+}_wx{t}_wy{p}_c{c}.ome.tif' \
  --segPattern  'p00{z}_x{x+}_y{y+}_wx{t}_wy{p}_c1.ome.tif' \
  --features    "BASIC_MORPHOLOGY,ALL_INTENSITY" \
  --kwargs      neighbor_distance=5 \
  --kwargs      pixels_per_micron=1.0 \
  --singleRoi   \
  --outDir      /data/features
```


## Options

This plugin takes seven input arguments and one output argument:

| Name               | Description                                                        | I/O    | Type          |
|--------------------|--------------------------------------------------------------------|--------|---------------|
| `--inpDir`         | Intensity images folder                                            | Input  | collection    |
| `--segDir`         | Label / segmentation images folder directory                                        | Input  | collection    |
| `--intPattern`     | Filepattern to parse intensity images                              | Input  | string        |
| `--segPattern`     | Filepattern to parse label images                                  | Input  | string        |
| `--features`       | [Feature groups or individual nyxus features (comma-separated or repeated)](https://pypi.org/project/nyxus/)                  | Input  | string        |      |
| `--singleRoi`      | Treat each intensity image as single ROI (whole-image features, no mask) | Input  | bool
| `--kwargs`      | Nyxus params as KEY=VALUE (repeatable; e.g., neighbor_distance=5) | Input  | list[str]         |
| `--outDir`         | Output collection                                                  | Output | collection    |
| `--preview`        | Generate a JSON file with outputs                                  | Output |
 JSON          |
