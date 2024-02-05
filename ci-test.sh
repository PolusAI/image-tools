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
  # # Ignore the directory if it contains the substring "polus-python-template"
  # if [[ "$dir" == *"polus-python-template"* ]]; then
  #   continue
  # fi
  # Ignore the directory if it contains any of the substrings in "ignored_dirs"
  for ignored_dir in $ignored_dirs; do
    if [[ "$dir" == *"$ignored_dir"* ]]; then
      continue 2
    fi
  done
  if [ -f "$dir/pyproject.toml" ]; then
    tools="$tools $dir"
  fi
done
# Remove leading and trailing spaces
tool_dirs=$(echo $tools | xargs)

# Print the directories that contain a "pyproject.toml" file
for tool in $tool_dirs; do
  echo "Found tool: $tool"
done

# For each tool directory, run the tests. Collect the tools that fail the tests
failed_tools=""
for tool in $tool_dirs; do
  echo "Running tests for $tool"
  cd $tool
  poetry install
  poetry run pytest -v || failed_tools="$failed_tools $tool"
  cd $current_dir
done

# Trim leading and trailing spaces
failed_tools=$(echo $failed_tools | xargs)

# If there are failed tools, print the names of the tools and exit with a non-zero status
if [ -n "$failed_tools" ]; then
  echo "The following tools failed the tests:"
  for tool in $failed_tools; do
    echo "$tool"
  done
  echo ""
  echo $failed_tools
  exit 1
fi

# # Failed tools were:
# failed_tools="/Users/najibishaq/Documents/axle/polusai/image-tools/visualization/ome-to-microjson /Users/najibishaq/Documents/axle/polusai/image-tools/visualization/microjson-to-ome /Users/najibishaq/Documents/axle/polusai/image-tools/formats/arrow-to-tabular-plugin /Users/najibishaq/Documents/axle/polusai/image-tools/formats/tabular-to-arrow-plugin /Users/najibishaq/Documents/axle/polusai/image-tools/formats/tabular-converter-plugin /Users/najibishaq/Documents/axle/polusai/image-tools/segmentation/mesmer-inference-plugin /Users/najibishaq/Documents/axle/polusai/image-tools/segmentation/cell-border-segmentation /Users/najibishaq/Documents/axle/polusai/image-tools/segmentation/mesmer-training-plugin /Users/najibishaq/Documents/axle/polusai/image-tools/features/feature-segmentation-eval /Users/najibishaq/Documents/axle/polusai/image-tools/features/region-segmentation-eval /Users/najibishaq/Documents/axle/polusai/image-tools/features/nyxus-plugin /Users/najibishaq/Documents/axle/polusai/image-tools/features/pixel-segmentation-eval /Users/najibishaq/Documents/axle/polusai/image-tools/clustering/k-means-clustering-plugin /Users/najibishaq/Documents/axle/polusai/image-tools/transforms/images/roi-relabel-plugin /Users/najibishaq/Documents/axle/polusai/image-tools/transforms/images/binary-operations-plugin /Users/najibishaq/Documents/axle/polusai/image-tools/transforms/images/image-assembler-plugin /Users/najibishaq/Documents/axle/polusai/image-tools/transforms/images/lumos-bleedthrough-correction-plugin /Users/najibishaq/Documents/axle/polusai/image-tools/transforms/tabular/tabular-thresholding-plugin /Users/najibishaq/Documents/axle/polusai/image-tools/transforms/tabular/tabular-merger-plugin /Users/najibishaq/Documents/axle/polusai/image-tools/regression/theia-bleedthrough-estimation-plugin"

# for tool in $failed_tools; do
#   # echo "Failed: $tool"
#   cd $tool
#   # If the directory does not contain a ".bumpversion.cfg" file, then exit
#   if [ ! -f ".bumpversion.cfg" ]; then
#     echo "Failed: $tool"
#     continue
#   fi
#   # Check if the "pyproject.toml" contains a line with the substring "tool.poetry.source"
#   if grep -q "commit = False" .bumpversion.cfg; then
#     # If it does, then print the tool name
#     echo "Failed: $tool"
#   fi
#   cd $current_dir
# done
