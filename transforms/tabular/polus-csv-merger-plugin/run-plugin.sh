#!/bin/bash

#!/bin/bash
version=$(<VERSION)
datapath=''

# Inputs
inpDir=/data/cytoplasm

# Output paths
outDir=/data/output

#Other params
stripExtension=false
dim=rows
sameRows= true

# Log level, must be one of ERROR, CRITICAL, WARNING, INFO, DEBUG
LOGLEVEL=INFO

docker run --mount type=bind,source=${datapath},target=/data/  \
            --env POLUS_LOG=${LOGLEVEL} \
            polusai/polus-csv-merger-plugin:${version} \
            --inpDir ${inpDir} \
            --stripExtension ${stripExtension} \
            --dim ${dim} \
            --outDir ${outDir}
            