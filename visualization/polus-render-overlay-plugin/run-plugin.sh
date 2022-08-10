#!/bin/bash

version=$(<VERSION)
datapath=$(readlink --canonicalize ../../data)
echo ${datapath}

# Inputs
stitchingVector=/data/eastman
filePattern="p02_x{x+}_y{y+}_wx{t}_wy{p}_c{c}.ome.tif"
concatenate=True

# Output paths
outDir=/data/eastman/overlays

docker run --mount type=bind,source=${datapath},target=/data/ \
            --user $(id -u):$(id -g) \
            polusai/render-overlay-plugin:${version} \
            --stitchingVector ${stitchingVector} \
            --filePattern ${filePattern} \
            --concatenate ${concatenate} \
            --outDir ${outDir}