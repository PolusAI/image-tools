# Idr Download (v0.1.0-dev0)

This tool enables the retrieval of data from the [idr](https://idr.openmicroscopy.org/) using the IDR web API.

## Note
In order to retrieve data from the IDR web server, users need to establish a VPN connection


Conda is employed to install all dependencies because one of the critical packages, `omero-py`, encountered installation issues with pip

Currently, the supported object types in a tool include:  `project`, `dataset`, `screen`, `plate`, `well`


## Building

To build the Docker image for the download plugin, run
`bash build-docker.sh`.

## Run the Docker image

To execute the built docker image for the download plugin, run
`bash run-plugin.sh`.

## Options

This plugin takes 4 input arguments and
1 output argument:

| Name            | Description                                                  | I/O    | Type        |
| --------------- | ------------------------------------------------------------ | ------ | ----------- |
| `--dataType`      | Object types to be retreived from Idr Server                    | Input  | String      |
| `--name  `      | Name of an object                   | Input  | String      |
| `--objectId  `      |  Identification of an object of an object                 | Input  | Integer      |
| `--outDir`      | Directory to store the downloaded data                  | Output | genericData |
| `--preview`      | Generate a JSON file with outputs                  | Output | JSON |



## Sample docker command:
```bash
docker run -v /home/ec2-user/data/:/home/ec2-user/data/ polusai/idr-download-tool:0.1.0-dev0 --dataType="plate" --name='plate1_1_013' --outDir=/home/ec2-user/data/output```
