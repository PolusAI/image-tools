#!/bin/bash

version=$(<VERSION)
echo "Running Bleed Through Estimation Plugin version: ${version}"

data_path=$(readlink -f ./data)
echo "Data path: ${data_path}"

docker run polusai/bleed-through-estimation-plugin:"${version}"

# Must be one of ERROR, CRITICAL, WARNING, INFO, DEBUG
POLUS_LOG=INFO

# Change to .ome.zarr to save output images as zarr files.
POLUS_IMG_EXT=".ome.tif"

# Inputs
inpDir=/data/input
filePattern="S1_R{r:d}_C1-C11_A1_y009_x009_c{c:ddd}.ome.tif"
groupBy="r"
channelOrdering="1,0,3,2,4,5,7,6,8,9"
selectionCriterion="MeanIntensity"
channelOverlap=1
kernelSize=5

# Output paths
outDir=/data/output

docker run --mount type=bind,source="${data_path}",target=/data/ \
            --user "$(id -u)":"$(id -g)" \
            --env POLUS_LOG="${POLUS_LOG}" \
            --env POLUS_IMG_EXT="${POLUS_IMG_EXT}" \
            polusai/bleed-through-estimation-plugin:"${version}" \
            --inpDir "${inpDir}" \
            --filePattern "${filePattern}" \
            --groupBy "${groupBy}" \
            --channelOrdering "${channelOrdering}" \
            --selectionCriterion "${selectionCriterion}" \
            --channelOverlap "${channelOverlap}" \
            --kernelSize "${kernelSize}" \
            --outDir "${outDir}"
