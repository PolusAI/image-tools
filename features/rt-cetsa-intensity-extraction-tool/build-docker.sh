#!/bin/bash

# Derive the tool name from the folder name
script_dir=$(dirname "$(realpath "$0")")
parent_folder=$(basename "$script_dir")
tool_name=$parent_folder

# The root of the repo
repo_root=$(git rev-parse --show-toplevel)

# Get the path to this tool from the repository root
tool_dir=$(python3 -c "import os.path; print(os.path.relpath('$script_dir', '$repo_root'))")

echo "Building docker image for tool: ${tool_name}"
echo "Tool path from root: ${tool_dir}"

version=$(<VERSION)

org=${DOCKER_ORG:-polusai}
tag="${org}/${tool_name}:${version}"
tag=$tag${ARCH_SUFFIX}

# Create a staging directory
mkdir -p ${repo_root}/docker_build
cd ${repo_root}/docker_build

# .gitignore is used as .dockerignore
cp ${repo_root}/.gitignore .dockerignore
# # TODO append tool_dir .gitignore to .dockerignore if it exists
cp -r ${repo_root}/segmentation/rt-cetsa-plate-extraction-tool .
cp -r ${repo_root}/${tool_dir} .

# build the docker image
build_cmd="build . -f ${repo_root}/${tool_dir}/Dockerfile --no-cache  -t ${tag} --build-arg TOOL_DIR=${tool_dir}"
echo "build docker image : $build_cmd"
docker $build_cmd

# # # clean up staging directory
rm -rf ${repo_root}/docker_build