#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/imagej-filter-tubeness-plugin:${version}