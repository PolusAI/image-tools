#!/bin/bash

version=$(<VERSION)
datapath=$(readlink --canonicalize ../data)

# Inputs
inpDir=/data/input

# Output paths
outDir=/data/output

docker run --mount type=bind,source=${datapath},target=/data/ \
            labshare/polus-ome-zarr-converter-plugin:${version} \
            --inpDir ${inpDir} \
            --outDir ${outDir} 
            