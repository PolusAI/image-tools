#!/bin/bash

#!/bin/bash
version=$(<VERSION)
#echo $(datapath)

# Inputs
inpDir=/data/csv_input

# Output paths
outDir=/data/csv_output

#Other params
stripExtension=false
dim=rows
# sameRows=

# Log level, must be one of ERROR, CRITICAL, WARNING, INFO, DEBUG
LOGLEVEL=INFO

docker run -v /Users/mezukn/Desktop/polus/data:/data \
            --env POLUS_LOG=${LOGLEVEL} \
            labshare/polus-csv-merger-plugin:${version} \
            --inpDir ${inpDir} \
            --stripExtension ${stripExtension} \
            --dim ${dim} \
            --outDir ${outDir}
            