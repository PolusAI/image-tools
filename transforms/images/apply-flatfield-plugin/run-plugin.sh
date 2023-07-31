#!/bin/bash

version=$(<VERSION)
datapath=$(readlink --canonicalize ./data)
echo ${datapath}

# Inputs
imgDir="/data/images"
imgPattern="p{p:d+}_x{x:d+}_y{y:d+}_wx{r:d+}_wy{z:d+}_c{c:d+}.ome.tif"
ffDir="/data/estimation"
ffPattern="p{p:d+}_x\\(01-24\\)_y\\(01-16\\)_wx\\(1-3\\)_wy\\(1-3\\)_c{c:d+}_flatfield.ome.tif"
dfPattern="p{p:d+}_x\\(01-24\\)_y\(01-16\\)_wx\\(1-3\\)_wy\\(1-3\\)_c{c:d+}_darkfield.ome.tif"
# photoPattern=""

# Output paths
outDir=/data/outputs

FILE_EXT=".ome.zarr"

docker run --mount type=bind,source=${datapath},target=/data/ \
            -e POLUS_EXT=${FILE_EXT} \
            --user $(id -u):$(id -g) \
            polusai/apply-flatfield-plugin:${version} \
            --imgDir ${imgDir} \
            --imgPattern ${imgPattern} \
            --ffDir ${ffDir} \
            --ffPattern ${ffPattern} \
            --ffPattern ${dfPattern} \
            --outDir ${outDir}
