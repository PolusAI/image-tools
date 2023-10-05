#!/bin/bash
version=$(<VERSION)

inpDir=/tmp/path/to/input
pyramidType=Zarr
filePattern="p02_x(01-24)_y(01-16)_wx(0-2)_wy(0-2)_c{c}.ome.tif"
imageType=image
outDir=/tmp/path/to/output
container_input_dir="/inpDir"
container_output_dir="/outDir"

docker run  -v $inpDir:/${container_input_dir} \
            -v $outDir:/${container_output_dir} \
            --user $(id -u):$(id -g) \
            polusai/precompute-slide-plugin:${version} \
            --inpDir ${inpDir} \
            --pyramidType ${pyramidType} \
            --filePattern ${filePattern} \
            --imageType ${imageType} \
            --outDir ${outDir}