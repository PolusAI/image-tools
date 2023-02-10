#!/bin/bash

version=$(<VERSION)
data_path=$(readlink --canonicalize ../../data)

# Must be one of ERROR, CRITICAL, WARNING, INFO, DEBUG
POLUS_LOG="INFO"

# Change to .ome.zarr to save output images as zarr files.
POLUS_EXT=".ome.tif"

# Inputs
inpDir=/data/input


# Output paths
outDir=/data/output

docker run --mount type=bind,source="${data_path}",target=/data/ \
            --user "$(id -u)":"$(id -g)" \
            --env POLUS_LOG="${POLUS_LOG}" \
            --env POLUS_EXT="${POLUS_EXT}" \
            polusai/zo1-segmentation-plugin:"${version}" \
            --inpDir ${inpDir} \
            --outDir ${outDir}
