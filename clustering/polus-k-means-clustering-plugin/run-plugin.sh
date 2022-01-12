#!/bin/bash

#!/bin/bash
version=$(<VERSION)
datapath=$(readlink --canonicalize ../data)

# Inputs
inpDir=/data/inputs

# Output paths
outDir=/data/outputs

#Additional args
methods=
minimumrange=
maximumrange=
numofclus=

# Log level, must be one of ERROR, CRITICAL, WARNING, INFO, DEBUG
LOGLEVEL=INFO

docker run --mount type=bind,source=${datapath},target=/data/  \
            --env POLUS_LOG=${LOGLEVEL} \
            polusai/feather-to-tabular-plugin:${version} \
            --inpDir ${inpDir} \
            --methods ${methods} \
            --minimumrange ${minimumrange} \
            --maximumrange ${maximumrange} \
            --numofclus ${numofclus} \
            --outDir ${outDir} \