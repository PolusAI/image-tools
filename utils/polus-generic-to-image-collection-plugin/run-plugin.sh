#!/bin/bash

version=$(<VERSION)
datapath=$(readlink --canonicalize ../data)

# Inputs
inpDir=/data/input

# Output paths
outDir=/data/output

docker run --mount type=bind,source=${datapath},target=/data/ \
            --user "$(id -u):$(id -g)" \
            polusai/generic-to-image-collection-plugin:${version} \
            --inpDir ${inpDir} \
            --outDir ${outDir} 
