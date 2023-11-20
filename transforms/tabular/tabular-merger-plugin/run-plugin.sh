#!/bin/bash
version=$(<VERSION)
datapath=$(readlink --canonicalize data)

# Inputs
inpDir=/data/input
filePattern=".*"

# Output paths
outDir=/data/output

#Other params
stripExtension=false
dim=rows
mapVar = "mask_intensity"

# Log level, must be one of ERROR, CRITICAL, WARNING, INFO, DEBUG
LOGLEVEL=INFO

docker run --mount type=bind,source=${datapath},target=/data/  \
            --env POLUS_LOG=${LOGLEVEL} \
            polusai/tabular-merger-plugin:${version} \
            --inpDir ${inpDir} \
            --filePattern ${filePattern} \
            --stripExtension ${stripExtension} \
            --dim ${dim} \
            --sameRows \
            --sameColumns \
            --mapVar ${mapVar} \
            --outDir ${outDir}
