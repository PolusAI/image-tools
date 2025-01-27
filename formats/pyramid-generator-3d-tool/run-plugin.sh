#!/bin/bash

version=$(<VERSION)

container_input_dir="/inpDir"
container_output_dir="/outDir"

# example usage 1, volume generation
subCmd="Vol"
inpDir=/tmp/path/to/input
filepattern="pattern"
groupBy="z"
outDir=/tmp/path/to/output
outImgName="output_image"

docker run -v $inpDir:/${container_input_dir} \
           -v $outDir:/${container_output_dir} \
            --user $(id -u):$(id -g) \
            polusai/pyramid-generator-3d-tool:${version} \
            --subCmd ${subCmd} \
            --inpDir ${container_input_dir} \
            --filePattern ${filepattern} \
            --groupBy ${groupBy} \
            --outDir ${container_output_dir} \
            --outImgName ${outImgName}

# example usage 2, 3d pyramid from zarr directory
subCmd="Py3D"
zarrDir=/tmp/path/to/zarr
outDir=${zarrDir} # output is saved in the same directory in this usage
baseScaleKey=0
numLevels=2

docker run -v $zarrDir:/${container_input_dir} \
            --user $(id -u):$(id -g) \
            polusai/pyramid-generator-3d-tool:${version} \
            --subCmd ${subCmd} \
            --zarrDir ${container_input_dir} \
            --outDir ${container_input_dir} \
            --baseScaleKey ${baseScaleKey} \
            --numLevels ${numLevels}

# example usage 3, 3d pyramid directly from image collection
subCmd="Py3D"
inpDir=/tmp/path/to/input
filepattern="pattern"
groupBy="z"
outDir=/tmp/path/to/output
outImgName="output_image"
baseScaleKey=0
numLevels=2

docker run -v $inpDir:/${container_input_dir} \
           -v $outDir:/${container_output_dir} \
            --user $(id -u):$(id -g) \
            polusai/pyramid-generator-3d-tool:${version} \
            --subCmd ${subCmd} \
            --inpDir ${container_input_dir} \
            --filePattern ${filepattern} \
            --groupBy ${groupBy} \
            --outDir ${container_output_dir} \
            --outImgName ${outImgName} \
            --baseScaleKey ${baseScaleKey} \
            --numLevels ${numLevels}
