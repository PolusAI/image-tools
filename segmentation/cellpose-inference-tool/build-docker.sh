#!/bin/bash

tool_dir="segmentation"
tool_name="cellpose-inference-tool"

version=$(<VERSION)
tag="polusai/${tool_name}:${version}"
echo "Building docker image with tag: ${tag}"

cur_dir=$(pwd)
repo_root=$(git rev-parse --show-toplevel)

cd ${repo_root}
cp ./${tool_dir}/${tool_name}/Dockerfile .
cp .gitignore .dockerignore
docker build . -t ${tag}
rm Dockerfile .dockerignore
cd ${cur_dir}
