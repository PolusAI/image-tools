#!/bin/bash

version=$(<VERSION)
data_path=$(readlink --canonicalize /data/axle/tests/cellpose_inference)

# Inputs
inpDir=/data/input
diameterMode=PixelSize
filePattern=".+"
pretrainedModel=nuclei

# GPU configuration for testing GPU usage in the container
GPUS=all

# Must be one of ERROR, CRITICAL, WARNING, INFO, DEBUG
POLUS_LOG=INFO

# Output paths
outDir=/data/output_zarr

# Remove the --gpus flag to test on CPU
docker run --mount type=bind,source="${data_path}",target=/data/ \
           --user "$(id -u)":"$(id -g)" \
           --gpus ${GPUS} \
           --env POLUS_LOG=${POLUS_LOG} \
           labshare/polus-cellpose-inference-plugin:"${version}" \
           --inpDir ${inpDir} \
           --diameterMode ${diameterMode} \
           --filePattern ${filePattern} \
           --pretrainedModel ${pretrainedModel} \
           --outDir ${outDir}