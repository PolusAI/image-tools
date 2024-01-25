#!/bin/bash

version=$(<VERSION)

# Update with your data
inpDir=/tmp/path/to/input
outDir=/tmp/path/to/output
filepattern="pattern"


container_input_dir="/inpDir"
container_output_dir="/outDir"

docker run -v $inpDir:/${container_input_dir} \
           -v $outDir:/${container_output_dir} \
            --user $(id -u):$(id -g) \
            {{cookiecutter.container_id}}:${version} \
            --inpDir ${container_input_dir} \
            --filePattern ${filepattern} \
            --outDir ${container_output_dir}
