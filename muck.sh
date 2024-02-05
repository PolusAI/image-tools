#!/bin/bash

ignored_dirs="polus-python-template ftl-label .venv"

# Get the current directory
current_dir=$(pwd)

# Get all directories in the current directory
dirs=$(find $current_dir -type d)
# Get all directories that have a "pyproject.toml" file
tools=""
for dir in $dirs; do
  # Ignore the current directory
  if [ "$dir" == "$current_dir" ]; then
    continue
  fi
  # Ignore the directory if it contains any of the substrings in "ignored_dirs"
  for ignored_dir in $ignored_dirs; do
    if [[ "$dir" == *"$ignored_dir"* ]]; then
      continue 2
    fi
  done
  # If the directory contains a "pyproject.toml" file, then add it to the list of tools
  if [ -f "$dir/pyproject.toml" ]; then
    tools="$tools $dir"
  fi
done
# Remove leading and trailing spaces
tool_dirs=$(echo $tools | xargs)

tool_names=""
for tool in $tool_dirs; do
  tool_name=$(basename $tool)
  tool_names="$tool_names $tool_name"
done
# Remove leading and trailing spaces
tool_names=$(echo $tool_names | xargs)
