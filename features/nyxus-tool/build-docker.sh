# #!/bin/bash

# Change the name of the tool here
tool_dir="features"
tool_name="nyxus-tool"

# The version is read from the VERSION file
version=$(<VERSION)
tag="polusai/${tool_name}:${version}"
echo "Building docker image with tag: ${tag}"

# The current directory and the repository root are saved in variables
cur_dir=$(pwd)
repo_root=$(git rev-parse --show-toplevel)

# Build directly from the project directory so COPY . picks up pyproject.toml
docker build --no-cache ${repo_root}/${tool_dir}/${tool_name} -t ${tag}

cd ${cur_dir}