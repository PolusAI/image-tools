#!/bin/bash

version=$(<VERSION)
datapath=$(readlink --canonicalize data)
echo ${datapath}

# Inputs
inp_dir=${datapath}/input
file_pattern=".*.ome.tif"
polygon_type='encoding'
out_dir=${datapath}/output

# #Show the help options
# #docker run polusai/segmentations-micojson-plugin:${version}

docker run -v ${datapath}:${datapath} \
            polusai/ome-to-microjson-plugin:${version} \
            --inpDir ${inp_dir} \
            --filePattern ${file_pattern} \
            --polygonType ${polygon_type} \
            --outDir ${out_dir}
