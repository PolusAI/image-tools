#!/bin/bash

version=$(<VERSION)
data_path=$(readlink --canonicalize /data/axle/tests)

# Inputs
inpDir=/data/output_zarr
filePattern=".+"
flowThreshold=1.0
cellprobThreshold=0.4

# Output paths
outDir=/data/output

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
            labshare/polus-vector-label-plugin:"${version}" \
            --inpDir ${inpDir} \
            --filePattern ${filePattern} \
            --flowThreshold ${flowThreshold} \
            --cellprobThreshold ${cellprobThreshold} \
            --outDir ${outDir}
