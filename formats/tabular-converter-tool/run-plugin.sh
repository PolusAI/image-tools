#!/bin/bash
version=$(<VERSION)
datapath=$(readlink --canonicalize ../../../data)

#Inputs
inpDir=/data/input

# Output paths
outDir=/data/output

# Input Fileformat
filePattern=".csv"

# Output Fileformat
fileExtension =".arrow"

#Show the help options
docker run polusai/tabular-converter-plugin:${version}

# Run the plugin
docker run --mount type=bind,source=${datapath},target=/data/ \
            polusai/tabular-converter-plugin:${version} \
            --inpDir ${inpDir} \
            --filePattern ${filePattern} \
            --fileExtension ${fileExtension} \
            --outDir ${outDir}
