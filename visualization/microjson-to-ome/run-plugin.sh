#!/bin/bash
version=$(<VERSION)
datapath=$(readlink --canonicalize data)
echo ${datapath}

# Inputs
inp_dir=${datapath}/input
file_pattern=".*.json"
out_dir=${datapath}/output

# #Show the help options
# docker run polusai/microjson-to-ome:${version}

docker run -v ${datapath}:${datapath} \
            polusai/microjson-to-ome:${version} \
            --inpDir ${inp_dir} \
            --filePattern ${file_pattern} \
            --outDir ${out_dir}
