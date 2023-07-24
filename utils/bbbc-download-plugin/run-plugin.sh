#!/bin/bash

version=$(<VERSION)
datapath=$(readlink --canonicalize data)
mkdir ${datapath}
# Inputs
name="BBBC001"

# # Output paths
outDir=${datapath}

# # Show the help options
# docker run polusai/bbbc-download-plugin:${version}

# # Run the plugin
docker run -v ${datapath}:${datapath} \
            polusai/bbbc-download-plugin:${version} \
            --name ${name} \
            --outDir ${outDir}
