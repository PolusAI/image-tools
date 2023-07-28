#!/bin/bash

version=$(<VERSION)
datapath=$(readlink --canonicalize data)
echo ${datapath}

# # Inputs
inp_dir=/data/input
file_pattern=".*"
dimension="384"
geometry_type="Polygon"
cell_width=2170
cell_height=2180
out_dir=/data/output


# # Show the help options
# # docker run polusai/render-overlay-plugin:${version}

docker run --mount type=bind,source=${datapath},target=/data/ \
            polusai/render-overlay-plugin:${version} \
            --inpDir ${inp_dir} \
            --filePattern ${file_pattern} \
            --dimension ${dimension} \
            --geometryType ${geometry_type} \
            --cellWidth ${cell_width} \
            --cellHeight ${cell_height} \
            --outDir ${out_dir}
