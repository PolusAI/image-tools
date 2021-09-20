#!/bin/bash

version=$(<VERSION)
data_path=$(readlink --canonicalize ../../../data/autocropping)

# Change to .ome.zarr to save output images as zarr files.
POLUS_EXT=".ome.tif"

# Must be one of ERROR, CRITICAL, WARNING, INFO, DEBUG
POLUS_LOG=INFO

# Inputs
inputDir=/data/input
filePattern="r{r}_c{c}.ome.tif"
groupBy="c"
cropX="true"
cropY="true"
cropZ="true"
smoothing="true"

# Outputs
outputDir=/data/output

docker run --mount type=bind,source="${data_path}",target=/data \
            --user "$(id -u)":"$(id -g)" \
            --env POLUS_EXT=${POLUS_EXT} \
            --env POLUS_LOG=${POLUS_LOG} \
            labshare/polus-autocropping-plugin:"${version}" \
            --inputDir ${inputDir} \
            --filePattern ${filePattern} \
            --groupBy ${groupBy} \
            --cropX ${cropX} \
            --cropY ${cropY} \
            --cropZ ${cropZ} \
            --smoothing ${smoothing} \
            --outputDir ${outputDir}
