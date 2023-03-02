#!/bin/bash

#!/bin/bash
version=$(<VERSION)
datapath=$(readlink --canonicalize ../data)

# Inputs
inpDir=/data/inputs


# Output paths
outDir=/data/outputs

# Output Fileformat
filePattern=".csv"

# Show the help options
docker run polusai/tabular-to-arrow-plugin:${version}

# Run the plugin
docker run -v /--mount type=bind,source=${datapath},target=/data/ \
            --env POLUS_LOG=${LOGLEVEL} \
            polusai/tabular-to-arrow-plugin:${version} \
            --inpDir ${inpDir} \
            --filePattern ${filePattern} \
            --outDir ${outDir} \
