#!/bin/bash

# Example run script for the Cellpose Inference Tool.
# Edit the variables below to match your data paths and desired parameters.
#
# GPU REQUIREMENTS (when USE_GPU=true):
#   - NVIDIA driver >= 525.60.13 (CUDA 12.0+)
#   - NVIDIA Container Toolkit installed on the host
#     https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
#   - Run Docker with --gpus all (or --gpus device=0 for a specific GPU)

version=$(<VERSION)

INP_DIR="/path/to/input/images"
OUT_DIR="/path/to/output/masks"
FILE_PATTERN=".+"
MODEL_TYPE="cyto3"
DIAMETER=0
MIN_SIZE=15
CHANNEL_CYTO=0
CHANNEL_NUC=-1
FLOW_THRESHOLD=0.4
CELLPROB_THRESHOLD=0.0
NITER=0
DO_3D=false
STITCH_THRESHOLD=0.0
ANISOTROPY=1.0
NO_NORM=false
NORM_PERCENTILE=""   # e.g. "1.0,99.0"  — leave empty to use default
BATCH_SIZE=8
AUGMENT=false
USE_GPU=true
EXCLUDE_ON_EDGES=false

docker run --gpus all \
  -e POLUS_IMG_EXT=".ome.tif" \
  -e NUM_WORKERS=1 \
  -e CUDA_VISIBLE_DEVICES=0 \
  -v "${INP_DIR}":"${INP_DIR}" \
  -v "${OUT_DIR}":"${OUT_DIR}" \
  "polusai/cellpose-inference-tool:${version}" \
  --inpDir="${INP_DIR}" \
  --filePattern="${FILE_PATTERN}" \
  --modelType="${MODEL_TYPE}" \
  --diameter="${DIAMETER}" \
  --minSize="${MIN_SIZE}" \
  --channelCyto="${CHANNEL_CYTO}" \
  --channelNuc="${CHANNEL_NUC}" \
  --flowThreshold="${FLOW_THRESHOLD}" \
  --cellprobThreshold="${CELLPROB_THRESHOLD}" \
  --niter="${NITER}" \
  --do3D="${DO_3D}" \
  --stitchThreshold="${STITCH_THRESHOLD}" \
  --anisotropy="${ANISOTROPY}" \
  ${NORM_PERCENTILE:+--normPercentile="${NORM_PERCENTILE}"} \
  --noNorm="${NO_NORM}" \
  --batchSize="${BATCH_SIZE}" \
  --augment="${AUGMENT}" \
  --useGpu="${USE_GPU}" \
  --excludeOnEdges="${EXCLUDE_ON_EDGES}" \
  --outDir="${OUT_DIR}"
