#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/cell-nuclei-segmentation-plugin:${version}