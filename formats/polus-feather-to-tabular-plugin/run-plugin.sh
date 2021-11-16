#!/bin/bash

#!/bin/bash
version=$(<VERSION)
#echo $(datapath)

# Inputs
inpDir=/data/feather

# Output paths
outDir=/data/outputs

#
filePattern=.parquet

# Log level, must be one of ERROR, CRITICAL, WARNING, INFO, DEBUG
LOGLEVEL=INFO

docker run -v /Users/mezukn/Desktop/polus/data:/data \
            --env POLUS_LOG=${LOGLEVEL} \
            polusai/feather-to-tabular-plugin:${version} \
            --inpDir ${inpDir} \
            --filePattern ${filePattern} \
            --outDir ${outDir} \
            
            