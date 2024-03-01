#!/bin/bash
version=$(<VERSION)
datapath=$(readlink --canonicalize data)

# Inputs
inpDir=/data/inputs
filePattern=".*_{row:c}{col:dd}_s{s:d}_w{channel:d}.*.tif"
outFilePattern="r01_x{row:c}_y{col:dd}_p{s:d}_c{channel:d}.ome.tif"
mapDirectory=true
# Output paths
outDir=/data/output

# Show the help options
docker run polusai/file-renaming-plugin:${version}

# Run the plugin
docker run --mount type=bind,source=${datapath},target=/data/ \
            polusai/file-renaming-plugin:${version} \
            --inpDir ${inpDir} \
            --filePattern ${filePattern} \
            --outFilePattern ${outFilePattern} \
            --mapDirectory
            --outDir ${outDir}
