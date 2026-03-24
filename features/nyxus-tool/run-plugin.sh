#!/bin/bash

version=$(<VERSION)
datapath=$(readlink --canonicalize data)

# CONFIGURATION
# Input / output paths **inside** the container
inp_dir=/data/intensity
seg_dir=/data/labels
out_dir=/data/features

# File patterns (adjust to match filenames)
int_pattern='p00{z}_x{x+}_y{y+}_wx{t}_wy{p}_c{c}.ome.tif'
seg_pattern='p00{z}_x{x+}_y{y+}_wx{t}_wy{p}_c1.ome.tif'

# Features – quote when using commas
features="BASIC_MORPHOLOGY,ALL_INTENSITY"

# Output format – now via environment variable
# Options: pandas (.csv), arrowipc (.arrow), parquet (.parquet)
export POLUS_TAB_EXT=".csv"

# Nyxus advanced parameters – now passed via --kwargs
kwargs=(
  "--kwargs" "neighbor_distance=5"
  "--kwargs" "pixels_per_micron=1.0"
  # Add more if needed, examples:
  # "--kwargs" "coarse_gray_depth=32"
  # "--kwargs" "ibsi=true"
)

# Show the help options
# docker run polusai/nyxus-plugin:${version}

echo "Running Nyxus plugin v${version}"
echo "Output format: ${POLUS_TAB_EXT}"
echo "Data mounted:  ${datapath} → /data"

docker run --rm \
  --mount type=bind,source="${datapath}",target=/data/ \
  --env POLUS_TAB_EXT="${POLUS_TAB_EXT}" \
  polusai/nyxus-tool:${version} \
    --inpDir       "${inp_dir}" \
    --segDir       "${seg_dir}" \
    --intPattern   "${int_pattern}" \
    --segPattern   "${seg_pattern}" \
    --features     "${features}" \
    "${kwargs[@]}" \
    --outDir       "${out_dir}"

echo "Done. Features should be in: ${datapath}/features/"
