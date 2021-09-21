#!/bin/bash

version=$(<VERSION)
datapath=$(readlink --canonicalize ../../../data)

# Inputs
primaryDir=/data/images/MaricRatBrain2019/standard/intensity
primaryPattern="S1_R1_C1-C11_A1_c00{c}.ome.tif"
operator="subtract"
secondaryDir=/data/output/images
secondaryPattern="S1_R1_C1-C11_A1_c00{c}.ome.tif"

# Output paths
outDir=/data/output/subtracted

# Log level, must be one of ERROR, CRITICAL, WARNING, INFO, DEBUG
LOGLEVEL=INFO

docker run --mount type=bind,source=${datapath},target=/data/ \
            --user $(id -u):$(id -g) \
            --env POLUS_LOG=${LOGLEVEL} \
            polusai/image-calculator-plugin:${version} \
            --primaryDir ${primaryDir} \
            --primaryPattern ${primaryPattern} \
            --operator ${operator} \
            --secondaryDir ${secondaryDir} \
            --secondaryPattern ${secondaryPattern} \
            --outDir ${outDir} 
            