#!/bin/bash

#!/bin/bash
version=$(<VERSION)
datapath='/Users/mezukn/Desktop/polus/s3/data/'

# Inputs
inpDir=/data/feather

# Output paths
outDir=/data/outputs

#
format=csv

# Log level, must be one of ERROR, CRITICAL, WARNING, INFO, DEBUG
LOGLEVEL=INFO

docker run --mount type=bind,source=${datapath},target=/data/  \
            --env POLUS_LOG=${LOGLEVEL} \
            polusai/feather-to-tabular-plugin:${version} \
            --inpDir ${inpDir} \
            --filePattern ${filePattern} \
            --outDir ${outDir} \
            
            