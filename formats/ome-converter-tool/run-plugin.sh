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
# docker run polusai/ome-converter-plugin:${version}

# Run the plugin
docker run -e POLUS_IMG_EXT=${fileExtension} --mount type=bind,source=${datapath},target=/data/ \
            polusai/ome-converter-tool:${version} \
            --inpDir ${inpDir} \
            --filePattern ${filePattern} \
            --outDir ${outDir}
