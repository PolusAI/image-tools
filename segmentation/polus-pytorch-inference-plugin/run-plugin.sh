#!/bin/bash

version=$(<VERSION)
data_path=$(readlink --canonicalize ../../../data/pytorch-inference)

# Inputs
modelDir=/data/model
imagesDir=/data/input
filePattern=".+"
device="gpu"

# Output paths
outDir=/data/output

# Remove the --gpus all to test on CPU
docker run --mount type=bind,source="${data_path}",target=/data \
            --user "$(id -u)":"$(id -g)" \
            --rm \
            --gpus "all" \
            --privileged -v /dev:/dev \
            polusai/pytorch-inference-plugin:"${version}" \
            --modelDir ${modelDir} \
            --imagesDir ${imagesDir} \
            --filePattern ${filePattern} \
            --device ${device} \
            --outDir ${outDir}
