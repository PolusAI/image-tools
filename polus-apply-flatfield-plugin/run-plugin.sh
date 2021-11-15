#!/bin/bash

version=$(<VERSION)
datapath=$(readlink --canonicalize ../data)
echo ${datapath}

# Inputs
imgDir=/data/input
imgPattern='r01c01f{p+}p0{z}-ch1sk1fk1fl1.ome.tif'
ffDir=/data/output_basic
brightPattern='r01c01f(001-121)p0{z}-ch1sk1fk1fl1_flatfield.ome.tif'
darkPattern='r01c01f(001-121)p0{z}-ch1sk1fk1fl1_darkfield.ome.tif'
# photoPattern=''

# Output paths
outDir=/data/output

docker run --mount type=bind,source=${datapath},target=/data/ \
            --user $(id -u):$(id -g) \
            labshare/polus-apply-flatfield-plugin:${version} \
            --imgDir ${imgDir} \
            --imgPattern ${imgPattern} \
            --ffDir ${ffDir} \
            --brightPattern ${brightPattern} \
            --darkPattern ${darkPattern} \
            --outDir ${outDir}
            # --photoPattern ${photoPattern} \
