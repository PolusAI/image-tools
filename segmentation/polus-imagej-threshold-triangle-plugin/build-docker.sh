#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/imagej-threshold-triangle-plugin:${version}