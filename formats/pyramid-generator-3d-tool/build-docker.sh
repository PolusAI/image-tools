#!/bin/bash

# Change the name of the tool here
tool_dir="formats/"
tool_name="pyramid-generator-3d-tool"

# The version is read from the VERSION file
version=$(<VERSION)
tag="polusai/${tool_name}:${version}"
echo "Building docker image with tag: ${tag}"

# The current directory and the repository root are saved in variables
cur_dir=$(pwd)
repo_root=$(git rev-parse --show-toplevel)
echo "Repository root: ${repo_root}"

# The Dockerfile and .dockerignore files are copied to the repository root before building the image
cd ${repo_root}
cp ./${tool_dir}/${tool_name}/Dockerfile .
cp .gitignore .dockerignore
docker build . -t ${tag}
rm Dockerfile .dockerignore
cd ${cur_dir}
