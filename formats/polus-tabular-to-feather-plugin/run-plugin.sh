#!/bin/bash

#!/bin/bash
version=$(<VERSION)
datapath=$(readlink --canonicalize ../data)

# Inputs
inpDir=/data/inputs


# Output paths
outDir=/data/feather

#
filePattern=.csv

# Log level, must be one of ERROR, CRITICAL, WARNING, INFO, DEBUG
LOGLEVEL=INFO

docker run -v /--mount type=bind,source=${datapath},target=/data/ \
            --env POLUS_LOG=${LOGLEVEL} \
            polusai/tabular-to-feather-plugin:${version} \
            --inpDir ${inpDir} \
            --filePattern ${filePattern} \
            --outDir ${outDir} \
            
            