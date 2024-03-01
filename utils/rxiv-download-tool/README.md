# Rxiv Download (v0.1.0-dev0)

This plugin allows to download data from open access archives. Currently this plugin supports downloading data from  [arxiv](https://www.openarchives.org/). Later additional support for other archives will be added.

## Building

To build the Docker image for the download plugin, run
`bash build-docker.sh`.

## Run the Docker image

To execute the built docker image for the download plugin, run
`bash run-plugin.sh`.

## Options

This plugin takes 2 input arguments and
1 output argument:

| Name            | Description                                                  | I/O    | Type        |
| --------------- | ------------------------------------------------------------ | ------ | ----------- |
| `--rxiv  `      | Download data from open access archives                    | Input  | String      |
| `--start  `      | Start date                   | Input  | String      |
| `--outDir`      | Directory to store the downloaded data                  | Output | genericData |
| `--preview`      | Generate a JSON file with outputs                  | Output | JSON |



## Sample docker command:
```docker run -v /home/ec2-user/data/:/home/ec2-user/data/ polusai/rxiv-download-tool:0.1.0-dev0 --rxiv="arXiv" --start='2023-2-16' --outDir=/home/ec2-user/data/output```
