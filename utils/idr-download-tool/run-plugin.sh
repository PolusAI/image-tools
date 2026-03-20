#!/bin/bash

version=$(<VERSION)
datapath=$(readlink --canonicalize data)
echo ${datapath}

# Inputs
out_dir=${datapath}/out
data_type="plate"
object_id=3139


# #Show the help options
# #docker run polusai/polusai/idr_download-tool:${version}

docker run -v ${datapath}:${datapath} \
            polusai/idr_download-tool:${version} \
            --dataType ${data_type} \
            --name ${name} \
            --objectId ${object_id} \
            --outDir ${out_dir}
