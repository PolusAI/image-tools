#!/bin/bash

version=$(<VERSION)
datapath=$(readlink --canonicalize /home/schaubnj/polus-data/polus/images/Eastman2021Infectivity)
echo ${datapath}

# Inputs
imgDir=/data/standard/intensity
imgPattern='p{p+}_x{x+}_y{y+}_wx{r+}_wy{z+}_c{c}.ome.tif'
ffDir=/data/basic
brightPattern='p{p+}_x(01-24)_y(01-16)_wx(1-3)_wy(1-3)_c{c}_flatfield.ome.tif'
darkPattern='p{p+}_x(01-24)_y(01-16)_wx(1-3)_wy(1-3)_c{c}_darkfield.ome.tif'
# photoPattern=''

# Output paths
outDir=/data/corrected

FILE_EXT=".ome.zarr"

docker run --mount type=bind,source=${datapath},target=/data/ \
            -e POLUS_EXT=${FILE_EXT} \
            --user $(id -u):$(id -g) \
            labshare/polus-apply-flatfield-plugin:${version} \
            --imgDir ${imgDir} \
            --imgPattern ${imgPattern} \
            --ffDir ${ffDir} \
            --brightPattern ${brightPattern} \
            --darkPattern ${darkPattern} \
            --outDir ${outDir}
