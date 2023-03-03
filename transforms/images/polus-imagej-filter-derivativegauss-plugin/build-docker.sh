#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/imagej-filter-derivativegauss-plugin:${version}