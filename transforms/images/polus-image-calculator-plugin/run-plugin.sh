#!/bin/bash

version=$(<VERSION)
datapath=$(readlink --canonicalize ../../../data)

# Inputs
primaryDir=/data/input
primaryPattern="S1_R1_C1-C11_A1_y0(00-14)_x0(00-21)_c0{cc}.ome.tif"
operator="subtract"
secondaryDir=/data/bleedthrough/images
secondaryPattern="original_S1_R1_C1-C11_A1_y0(00-14)_x0(00-21)_c0{cc}.ome.tif"

# Output paths
outDir=/data/output

# Log level, must be one of ERROR, CRITICAL, WARNING, INFO, DEBUG
LOGLEVEL=INFO

docker run --mount type=bind,source=${datapath},target=/data/ \
            --user $(id -u):$(id -g) \
            --env POLUS_LOG=${LOGLEVEL} \
            labshare/polus-image-calculator-plugin:${version} \
            --primaryDir ${primaryDir} \
            --primaryPattern ${primaryPattern} \
            --operator ${operator} \
            --secondaryDir ${secondaryDir} \
            --secondaryPattern ${secondaryPattern} \
            --outDir ${outDir} 
            