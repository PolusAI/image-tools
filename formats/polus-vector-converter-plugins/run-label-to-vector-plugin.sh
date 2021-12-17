#!/bin/bash

version=$(<VERSION)
# data_path=$(readlink --canonicalize /home/schaubnj/anaconda3/envs/py39/lib/python3.9/site-packages/polus/images/Nadia2017ImportTest/raw/labels_columbus)
data_path=/home/schaubnj/Desktop/Projects/polus-plugins/test

# Inputs
inpDir=/data/3d_labels
filePattern=".+"

# Output paths
outDir=/data/3d_vectors

# GPU configuration for testing GPU usage in the container
GPUS=all

# Must be one of ERROR, CRITICAL, WARNING, INFO, DEBUG
POLUS_LOG=INFO

# Change to .ome.zarr to save output images as zarr files.
POLUS_EXT=".ome.tif"

# If your computer does not have a gpu, you need to remove the line with the --gpu flag.
docker run --mount type=bind,source="${data_path}",target=/data/ \
            --user "$(id -u)":"$(id -g)" \
            --gpus ${GPUS} \
            --env POLUS_LOG=${POLUS_LOG} \
            --env POLUS_EXT=${POLUS_EXT} \
            polusai/label-to-vector-plugin:"${version}" \
            --inpDir ${inpDir} \
            --filePattern ${filePattern} \
            --outDir ${outDir}
