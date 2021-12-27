#!/bin/bash

version=$(<VERSION)
datapath=$(readlink --canonicalize ../../data)

# Inputs
inpDir=/data/images/MaricRatBrain2019/standard/intensity
# convert from ome.tif to ome.zarr
filePattern="S1_R1_C1-C11_A1_c0{c}0.ome.tif"
fileExtension = ".ome.tif" 
# convert from ome.zarr to ome.tif
# filePattern="S1_R1_C1-C11_A1_c0{c}0.ome.zarr"
# fileExtension=".ome.tif"
# Output paths
outDir=/data/output

docker run --mount type=bind,source=${datapath},target=/data/ \
            labshare/polus-ome-converter-plugin:${version} \
            --inpDir ${inpDir} \
            --filePattern ${filePattern} \
            --fileExtension ${fileExtension} \
            --outDir ${outDir} 
            