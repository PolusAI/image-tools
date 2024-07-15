
#!/bin/bash

# Change the name of the tool here
tool_dir="regression"
tool_name="basic-flatfield-estimation-tool"

# The version is read from the VERSION file
version=$(<VERSION)
tag="polusai/${tool_name}:${version}"
echo "Building docker image with tag: ${tag}"

# The current directory and the repository root are saved in variables
cur_dir=$(pwd)
repo_root=$(git rev-parse --show-toplevel)

# The Dockerfile and .dockerignore files are copied to the repository root before building the image
cd ${repo_root}
cp ./${tool_dir}/${tool_name}/Dockerfile .
cp .gitingore .dockerignore
docker build . -t ${tag}
rm Dockerfile .dockerignore
cd ${cur_dir}
