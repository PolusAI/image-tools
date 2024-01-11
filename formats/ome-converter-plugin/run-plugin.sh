#!/bin/bash

version=$(<VERSION)
datapath=$(readlink --canonicalize data)

# Inputs
inpDir=/data/input
filePattern=".*"
fileExtension=".ome.zarr"

# Output paths
outDir=/data/output

# Show the help options
docker run polusai/ome-converter-plugin:${version}

# Run the plugin
docker run --mount type=bind,source=${datapath},target=/data/ \
            polusai/ome-converter-plugin:${version} \
            --inpDir ${inpDir} \
            --filePattern ${filePattern} \
            --fileExtension ${fileExtension} \
            --outDir ${outDir}
