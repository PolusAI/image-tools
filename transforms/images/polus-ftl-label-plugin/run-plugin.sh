#!/bin/bash

version=$(<VERSION)
data_path=$(readlink --canonicalize ../../../../data/ftl-label)

# Must be one of ERROR, CRITICAL, WARNING, INFO, DEBUG
POLUS_LOG="INFO"

# Change to .ome.zarr to save output images as zarr files.
POLUS_IMG_EXT=".ome.tif"

# Inputs
inpDir=/data/input-2d
connectivity=1
binarizationThreshold=0.5

# Output paths
outDir=/data/output

docker run --mount type=bind,source="${data_path}",target=/data/ \
            --user "$(id -u)":"$(id -g)" \
            --env POLUS_LOG="${POLUS_LOG}" \
            --env POLUS_IMG_EXT="${POLUS_IMG_EXT}" \
            polusaiftl-label-plugin:"${version}" \
            --inpDir ${inpDir} \
            --connectivity ${connectivity} \
            --binarizationThreshold ${binarizationThreshold} \
            --outDir ${outDir}
