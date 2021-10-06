#!/bin/bash

#!/bin/bash
version=$(<VERSION)
#echo $(datapath)

# Inputs
inpDir=/opt/executables/data/inputs


# Output paths
outDir=/opt/executables/data/outputs

# Log level, must be one of ERROR, CRITICAL, WARNING, INFO, DEBUG
LOGLEVEL=INFO

docker run -v /Users/mezukn/polus-plugins/polus-tabular-to-feather-plugin/src/data:/opt/executables/data \
            --env POLUS_LOG=${LOGLEVEL} \
            polusai/tabular-to-feather-plugin:${version} \
            --inpDir ${inpDir} \
            --outDir ${outDir} 
            