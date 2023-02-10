#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/imagej-filter-dog-plugin:${version}