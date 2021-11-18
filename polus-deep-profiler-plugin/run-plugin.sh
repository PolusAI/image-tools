#!/bin/bash

version=$(<VERSION)
datapath=$(readlink --canonicalize ../data)

# Inputs
inputDir=/data/path_to_files
maskDir=/data/path_to_mask
featureDir=/data/path_to_featureDir
model='VGG16'
batchSize=8


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
            labshare/polus-deep-profiler-plugin:${version} \
            --inputDir ${inputDir} \
            --maskDir ${maskDir} \
            --featureDir ${featureDir} \
            --model ${model} \
            --batchSize ${batchSize} \
            --outDir ${outDir} 
            