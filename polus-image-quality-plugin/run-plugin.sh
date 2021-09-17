#!/bin/bash

version=$(<VERSION)
datapath=$(readlink --canonicalize ../data)

# Inputs
inputDir=/data/path_to_files
minI="minI"
maxI="maxI"
scale="scale"
filename="filename"

# Output paths
outDir=/data/path_to_output

# GPU configuration for testing GPU usage in the container
GPUS=all

# Log level, must be one of ERROR, CRITICAL, WARNING, INFO, DEBUG
LOGLEVEL=INFO

docker run --mount type=bind,source=${datapath},target=/data/ \
            --user $(id -u):$(id -g) \
            --gpus ${GPUS} \
            --env POLUS_LOG=${LOGLEVEL} \
            labshare/polus-image-quality-plugin:${version} \
            --inputDir ${inputDir} \
            --minI ${minI} \
            --maxI ${maxI} \
            --scale ${scale} \
            --filename ${filename} \
            --outDir ${outDir} 
            