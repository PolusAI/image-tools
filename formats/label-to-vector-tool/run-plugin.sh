#!/bin/bash

version=$(<VERSION)
echo "Version: ${version}"

data_path=$(readlink -f ./data)
echo "Data path: ${data_path}"

docker run polusai/label-to-vector-plugin:"${version}"

# Inputs
inpDir=/data/input
filePattern=".*"

# Output paths
outDir=/data/output

# GPU configuration for testing GPU usage in the container
GPUS=all

# Must be one of ERROR, CRITICAL, WARNING, INFO, DEBUG
POLUS_LOG=INFO

# Change to .ome.zarr to save output images as zarr files.
POLUS_IMG_EXT=".ome.tif"

# If your computer does not have a gpu, you need to remove the line with the --gpu flag.
            # --gpus ${GPUS} \
docker run --mount type=bind,source="${data_path}",target=/data/ \
            --user "$(id -u)":"$(id -g)" \
            --env POLUS_LOG=${POLUS_LOG} \
            --env POLUS_IMG_EXT=${POLUS_IMG_EXT} \
            polusai/label-to-vector-plugin:"${version}" \
            --inpDir ${inpDir} \
            --filePattern ${filePattern} \
            --outDir ${outDir}
