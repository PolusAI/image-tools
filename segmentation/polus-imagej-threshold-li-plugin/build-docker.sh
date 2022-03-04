#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/imagej-threshold-li-plugin:${version}