#!/bin/bash

version=$(<VERSION)
data_path=$(readlink --canonicalize ../../../data/bleed-through-estimation)

# Must be one of ERROR, CRITICAL, WARNING, INFO, DEBUG
POLUS_LOG=INFO

# Change to .ome.zarr to save output images as zarr files.
POLUS_EXT=".ome.tif"

# Inputs
inpDir=/data/rat-brain-input
filePattern="S1_R{r}_C1-C11_A1_y009_x009_c0{cc}.ome.tif"
groupBy="c"
channelOrdering="1,0,3,2,4,5,7,6,8,9"

# Output paths
outDir=/data/rat-brain-output/images
csvDir=/data/rat-brain-output/csvs

docker run --mount type=bind,source="${data_path}",target=/data/ \
            --user "$(id -u)":"$(id -g)" \
            --env POLUS_LOG="${POLUS_LOG}" \
            --env POLUS_EXT="${POLUS_EXT}" \
            polusai/bleed-through-estimation-plugin:"${version}" \
            --inpDir "${inpDir}" \
            --filePattern "${filePattern}" \
            --groupBy "${groupBy}" \
            --channelOrdering "${channelOrdering}" \
            --outDir "${outDir}" \
            --csvDir "${csvDir}"
