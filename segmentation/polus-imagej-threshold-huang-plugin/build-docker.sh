#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/imagej-threshold-huang-plugin:${version}