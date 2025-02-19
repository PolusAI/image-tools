#!/bin/bash

version=$(<VERSION)
datapath=$(readlink --canonicalize data)
echo ${datapath}

# # Inputs
inp_dir=${datapath}/input
stitch_dir=${datapath}/stitchvector
file_pattern=".*.csv"
stitch_pattern='x{x:dd}_y{y:dd}_c{c:d}.ome.tif'
geometry_type="Polygon"
group_by=None
out_dir=${datapath}/output



docker run -v ${datapath}:${datapath} \
            polusai/tabular-to-microjson-tool:${version} \
            --inpDir ${inp_dir} \
            --stitchDir ${stitch_dir} \
            --filePattern ${file_pattern} \
            --stitchPattern ${stitch_pattern} \
            --groupBy ${group_by} \
            --geometryType ${geometry_type} \
            --tileJson \
            --outDir ${out_dir}
