#!/bin/bash

version=$(<VERSION)
datapath=$(readlink --canonicalize ../../data)

# Inputs
inpDir=/data/images/MaricRatBrain2019/standard/intensity
filePattern="S1_R1_C1-C11_A1_y{y+}_x{x+}_c{c+}.ome.tif"

# Output paths
outDir=/data/output

docker run --mount type=bind,source=${datapath},target=/data/ \
            polusai/ome-zarr-converter:${version} \
            --inpDir ${inpDir} \
            --filePattern ${filePattern} \
            --outDir ${outDir} 
            