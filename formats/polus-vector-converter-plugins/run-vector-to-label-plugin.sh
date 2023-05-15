#!/bin/bash

version=$(<VERSION)
data_path=$(readlink --canonicalize ../../../data)

# Inputs
inpDir=/data/cellpose_inference/rat-brain-standard-out
filePattern="S1_R{r}_C1-C11_A1_c000_flow.ome.zarr"
flowMagnitudeThreshold=0.1

# Output paths
outDir=/data/vector_converters/vector_to_label/rat-brain-standard-out

# Must be one of ERROR, CRITICAL, WARNING, INFO, DEBUG
POLUS_LOG=INFO

# Change to .ome.zarr to save output images as zarr files.
POLUS_EXT=".ome.tif"

# If your computer does not have a gpu, you need to remove the line with the --gpu flag.
docker run --mount type=bind,source="${data_path}",target=/data/ \
            --user "$(id -u)":"$(id -g)" \
            --env POLUS_LOG=${POLUS_LOG} \
            --env POLUS_EXT=${POLUS_EXT} \
            polusai/vector-to-label-plugin:"${version}" \
            --inpDir ${inpDir} \
            --filePattern ${filePattern} \
            --flowMagnitudeThreshold ${flowMagnitudeThreshold} \
            --outDir ${outDir}
