#!/bin/bash

version=$(<VERSION)
datapath=$(readlink --canonicalize data)

# Inputs
name="BBBC001"

# Output paths
outDir=/data/output

# Show the help options
#docker run polusai/bbbc-download-plugin:${version}

# Run the plugin
docker run --mount type=bind,source=${datapath},target=/data/ \
            polusai/bbbc-download-plugin:${version} \
            --name ${name} \
            --outDir ${outDir}