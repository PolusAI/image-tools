#!/bin/bash
version=$(<VERSION)
datapath=$(readlink --canonicalize ../../../data)

#Inputs
inpDir=/data/input

# Output paths
outDir=/data/output

# Output Fileformat
filePattern=".fcs"

#Show the help options
docker run polusai/tabular-converter-plugin:${version}

# Run the plugin
docker run --mount type=bind,source=${datapath},target=/data/ \
            polusai/tabular-converter-plugin:${version} \
            --inpDir ${inpDir} \
            --filePattern ${filePattern} \
            --outDir ${outDir}