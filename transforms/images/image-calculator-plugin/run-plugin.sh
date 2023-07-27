#!/bin/bash

version=$(<VERSION)
echo "Version: ${version}"

datapath=$(readlink -f ./data)
echo "Data path: ${datapath}"

docker run polusai/image-calculator-plugin:"${version}"

# Inputs
primaryDir=/data/primary
primaryPattern="S1_R1_C1-C11_A1_c00{c}.ome.tif"
operator="subtract"
secondaryDir=/data/secondary
secondaryPattern="S1_R1_C1-C11_A1_c00{c}.ome.tif"

# Output paths
outDir=/data/output

# Log level, must be one of ERROR, CRITICAL, WARNING, INFO, DEBUG
LOG_LEVEL=INFO

docker run --mount type=bind,source=${datapath},target=/data/ \
            --user $(id -u):$(id -g) \
            --env POLUS_LOG=${LOG_LEVEL} \
            polusai/image-calculator-plugin:${version} \
            --primaryDir ${primaryDir} \
            --primaryPattern ${primaryPattern} \
            --operator ${operator} \
            --secondaryDir ${secondaryDir} \
            --secondaryPattern ${secondaryPattern} \
            --outDir ${outDir}
