#!/bin/bash

version=$(<VERSION)
datapath=$(readlink --canonicalize data)
echo ${datapath}

# # Inputs
inp_dir=${datapath}/input
file_pattern=".*.csv"
dimensions="384"
geometry_type="Polygon"
cell_width=2170
cell_height=2180
out_dir=${datapath}/output


# # # Show the help options
# # # docker run polusai/tabular-to-microjson-plugin:${version}

docker run -v ${datapath}:${datapath} \
            polusai/tabular-to-microjson-plugin:${version} \
            --inpDir ${inp_dir} \
            --filePattern ${file_pattern} \
            --dimensions ${dimensions} \
            --geometryType ${geometry_type} \
            --cellWidth ${cell_width} \
            --cellHeight ${cell_height} \
            --outDir ${out_dir}
