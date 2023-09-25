#!/bin/bash

version=$(<VERSION)
datapath=$(readlink --canonicalize data)

# Inputs
gt_dir=${datapath}/gt_dir
pred_dir=${datapath}/pred_dir
file_pattern=".*.csv"
out_dir=${datapath}/output

# #Show the help options
# #docker run polusai/feature-evaluation-plugin:${version}

docker run -v ${datapath}:${datapath} \
            polusai/feature-evaluation-plugin:${version} \
            --GTDir ${gt_dir} \
            --PredDir ${pred_dir} \
            --filePattern ${file_pattern} \
            --combineLabels \
            --singleOutFile \
            --outDir ${out_dir}
