#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/imagej-filter-gauss-plugin:${version}