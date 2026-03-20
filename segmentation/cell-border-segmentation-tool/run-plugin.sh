#!/bin/bash

version=$(<VERSION)
datapath=$(readlink --canonicalize data)

# Must be one of ERROR, CRITICAL, WARNING, INFO, DEBUG
POLUS_LOG="INFO"

# Change to .ome.zarr to save output images as zarr files.
POLUS_EXT=".ome.tif"

# Inputs
inpDir=/data/input
filePattern=".*.ome.tif"

# Output paths
outDir=/data/output

# #Show the help options
# #docker run polusai/zo1-border-segmentation-plugin:${version}

docker run -v ${datapath}:${datapath} \
            polusai/cell-border-segmentation-plugin:${version} \
            --inpDir ${inp_dir} \
            --filePattern ${file_pattern} \
            --outDir ${out_dir}
