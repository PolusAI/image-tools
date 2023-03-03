#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/imagej-filter-partialderivative-plugin:${version}