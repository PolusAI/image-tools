#!/bin/bash

version=$(<VERSION)

data_path=$(readlink --canonicalize ./data)

# Change to .ome.zarr to save output images as zarr files.
POLUS_EXT=".ome.tif"

# Must be one of ERROR, CRITICAL, WARNING, INFO, DEBUG
POLUS_LOG=INFO

# Inputs
inpDir=/data/input
method="randomize"

# Outputs
outDir=/data/output

docker run --mount type=bind,source="${data_path}",target=/data \
            --user "$(id -u)":"$(id -g)" \
            --env POLUS_EXT=${POLUS_EXT} \
            --env POLUS_LOG=${POLUS_LOG} \
            polusai/roi-relabel-plugin:"${version}" \
            --inpDir ${inpDir} \
            --method ${method} \
            --outDir ${outDir}
