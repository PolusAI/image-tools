#!/bin/bash

version=$(<VERSION)
datapath=$(readlink --canonicalize ../data)

# Inputs
inpDir=/data/path_to_images
segDir=/data/path_to_label_images
filePattern='p{p+}.*.ome.tif'
features="BASIC_MORPHOLOGY","ALL_INTENSITY"
# More details available at https://github.com/PolusAI/nyxus
neighborDist=5.0
pixelPerMicron=1.0
outDir=/data/path_to_output


# Log level, must be one of ERROR, CRITICAL, WARNING, INFO, DEBUG
LOGLEVEL=INFO
docker run --mount type=bind,source=${datapath},target=/data/ \
            --user $(id -u):$(id -g) \
            --env POLUS_LOG=${LOGLEVEL} \
            polusai/scaled-nyxus:${version} \
            --inpDir ${inpDir} \
            --segDir ${segDir} \
            --outDir ${outDir} \
            --filePattern ${filePattern} \
            --features ${features} \
            --neighborDist ${neighborDist} \
            --pixelPerMicron ${pixelPerMicron} \