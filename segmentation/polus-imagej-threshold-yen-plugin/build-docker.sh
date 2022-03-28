#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/imagej-threshold-yen-plugin:${version}
