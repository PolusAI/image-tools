#!/bin/bash

#!/bin/bash
version=$(<VERSION)
#echo $(datapath)

# Inputs
inpDir=/data/inputs


# Output paths
outDir=/data/feather

#
filePattern=.csv

# Log level, must be one of ERROR, CRITICAL, WARNING, INFO, DEBUG
LOGLEVEL=INFO

docker run -v /Users/mezukn/Desktop/polus/data:/data \
            --env POLUS_LOG=${LOGLEVEL} \
            polusai/tabular-to-feather-plugin:${version} \
            --inpDir ${inpDir} \
            --filePattern ${filePattern} \
            --outDir ${outDir} \
            
            