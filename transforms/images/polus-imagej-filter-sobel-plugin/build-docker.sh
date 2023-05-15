#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/imagej-sobel-plugin:${version}