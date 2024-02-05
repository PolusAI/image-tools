#!/bin/bash

#!/bin/bash
version=$(<VERSION)
datapath=$(readlink --canonicalize data)

# Inputs
inpDir=/data/input

# Output paths
outDir=/data/output

# Output tabular file format
fileFormat='.csv'

# Show the help options
docker run polusai/arrow-to-tabular-plugin:${version}

# Run the plugin
docker run --mount type=bind,source=${datapath},target=/data/  \
            --env POLUS_LOG=${LOGLEVEL} \
            polusai/arrow-to-tabular-plugin:${version} \
            --inpDir ${inpDir} \
            --fileFormat ${fileFormat} \
            --outDir ${outDir}
