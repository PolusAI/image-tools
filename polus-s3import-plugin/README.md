# Polus S3 Import Plugin

This WIPP plugin will import data from a folder in an S3 bucket. Once the data is downloaded, it is made into an image collection. Files in an S3 bucket are checked against a list of supported Bioformats extensions and imported. If the import metadata option is selected, then formats not supported by Bioformats are imported. Files inside of subfolders will not be downloaded.

For more information on WIPP, visit the [official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).

For more information on Bioformats, visit the [official Bioformats page](https://www.openmicroscopy.org/bio-formats/). This plugin does not directly use Bioformats.

_Note:_ The AWS access key and secret are visible in the argo logs. Do not use this plugin to access a close repository if WIPP is deployed on an insecure system.

## Build the plugin

To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

To update the list of supported files for newer versions of Bioformats:

1. Generate the list of supported file types (using the [formatlist tool](https://docs.openmicroscopy.org/bio-formats/6.0.1/users/comlinetools/index.html))
2. Run `generateBFList.py`
3. Build the docker image using `./build-docker.sh`

## Install WIPP Plugin

In WIPP, navigate to the plugins page and add a new plugin. Paste the contents of `plugin.json` into the pop-up window and submit.

## Options

This plugin takes 5 input arguments and 1 output argument:

| Name                | Description                         | I/O    | Type |
|---------------------|-------------------------------------|--------|------|
| `--s3Bucket`        | Name of S3 bucket                   | Input  | String |
| `--s3Key`           | Path to data inside bucket          | Input  | String |
| `--awsAccessKey`    | AWS access key id                   | Input  | String |
| `--awsAccessSecret` | AWS secret access key               | Input  | String |
| `--getMeta`         | If true, will import metadata files | Input  | Boolean |
| `--outDir`          | Output image collection             | Output | String |

## Run the plugin

### Run the Docker Container

```bash
docker run -v /path/to/data:/data polus-s3import-plugin \
  --s3Bucket "Name of bucket" \
  --s3Key "Path/To/Data/" \
  --awsAccessKey AWSAccessKeyId \
  --awsSecretKey AWSSecretKey \
  --outDir /data/output
  --getMeta False
```
