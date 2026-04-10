#!/bin/bash

version=$(<VERSION)
datapath=$(readlink --canonicalize data)
echo ${datapath}

# Inputs
int_dir=${datapath}/input
seg_dir=${datapath}/segmentations
file_pattern="x{x:d+}_y{y:d+}_p{p:d+}_c{c:d+}.ome.tif"
polygon_type='encoding'
features="MEAN,MEDIAN"
neighbor_dist=5
pixel_per_micron=1.0
out_dir=${datapath}/output


docker run -v ${datapath}:${datapath} \
            polusai/ome-to-microjson-tool:${version} \
            --intDir ${inp_dir} \
            --segDir ${seg_dir} \
            --filePattern ${file_pattern} \
            --polygonType ${polygon_type} \
            --features ${features} \
            --neighborDist ${neighbor_dist} \
            --pixelPerMicron ${pixel_per_micron} \
            --outDir ${out_dir} \
            --tileJson
