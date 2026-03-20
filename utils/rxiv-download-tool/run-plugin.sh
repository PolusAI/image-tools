#!/bin/bash

version=$(<VERSION)
datapath=$(readlink --canonicalize data)
echo ${datapath}

# Inputs
out_dir=${datapath}/out
rxiv="arXiv"
start="2023-12-16"


# #Show the help options
# #docker run polusai/rxiv-download-plugin:${version}

docker run -v ${datapath}:${datapath} \
            polusai/rxiv-download-plugin:${version} \
            --rxiv ${rxiv} \
            --start ${start} \
            --outDir ${out_dir}
