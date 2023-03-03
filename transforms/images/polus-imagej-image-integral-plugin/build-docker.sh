#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/imagej-image-integral-plugin:${version}