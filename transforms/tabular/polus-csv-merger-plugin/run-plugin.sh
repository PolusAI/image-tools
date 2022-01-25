#!/bin/bash

#!/bin/bash
version=$(<VERSION)
echo $(datapath)

# Inputs
inpDir=/data/input

# Output paths
outDir=/data/output

#Other params
stripExtension=false
dim=columns
# sameRows=True

# Log level, must be one of ERROR, CRITICAL, WARNING, INFO, DEBUG
LOGLEVEL=INFO

docker run --mount type=bind,source=${datapath},target=/data/  \
            --env POLUS_LOG=${LOGLEVEL} \
            labshare/polus-csv-merger-plugin:${version} \
            --inpDir ${inpDir} \
            --stripExtension ${stripExtension} \
            --dim ${dim} \
            --outDir ${outDir}
            