#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/imagej-threshold-apply-plugin:${version}