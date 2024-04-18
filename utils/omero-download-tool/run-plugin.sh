#!/bin/bash

version=$(<VERSION)
datapath=$(readlink --canonicalize data)
echo ${datapath}

# Inputs
out_dir=${datapath}/out
dataType="well"
id=1084


# #Show the help options
# #docker run polusai/omero-download-tool:${version}

docker run -v ${datapath}:${datapath} \
            polusai/omero-download-tool:${version} \
            --dataType ${dataType} \
            --name ${name} \
            --id ${name} \
            --outDir ${out_dir}
