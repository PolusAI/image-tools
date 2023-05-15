#!/bin/bash

version=$(<VERSION)
datapath=$(readlink --canonicalize data)

# Inputs
inp_dir=/data/path_to_images
seg_dir=/data/path_to_label_images
int_pattern='p00{z}_x{x+}_y{y+}_wx{t}_wy{p}_c{c}.ome.tif'
seg_pattern='p00{z}_x{x+}_y{y+}_wx{t}_wy{p}_c1.ome.tif'
features="BASIC_MORPHOLOGY","ALL_INTENSITY"
file_extension = ".csv"
# More details available at https://github.com/PolusAI/nyxus
neighbor_dist=5.0
pixel_per_micron=1.0
out_dir=/data/path_to_output


# Show the help options
# docker run polusai/nyxus-plugin:${version}

docker run --mount type=bind,source=${datapath},target=/data/ \
            polusai/nyxus-plugin:${version} \
            --inpDir ${inp_dir} \
            --segDir ${seg_dir} \
            --outDir ${out_dir} \
            --intPattern ${int_pattern} \
            --segPattern ${seg_pattern} \
            --features ${features} \
            --fileExtension ${file_extension} \
            --neighborDist ${neighbor_dist} \
            --pixelPerMicron ${pixel_per_micron} \
