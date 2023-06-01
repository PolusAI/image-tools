#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/imagej-image-invert-plugin:${version}