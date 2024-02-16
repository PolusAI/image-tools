#!/bin/bash

version=$(<VERSION)
datapath=$(readlink --canonicalize ../../../data)

# Inputs
filePattern="r00{r}_z000_y0{yy}_x0{xx}_c0{cc}.ome.tif"
inpDir=/data/input
layout=xy,r

# Optional Inputs
imageSpacing=100
gridSpacing=10

# Output paths
outDir=/data/output

# Show the help options
docker run polusai/montage-plugin:${version}

# Run the plugin
docker run --mount type=bind,source=${datapath},target=/data/ \
            polusai/montage-plugin:${version} \
            --filePattern ${filePattern} \
            --inpDir ${inpDir} \
            --layout ${layout} \
            --outDir ${outDir}
