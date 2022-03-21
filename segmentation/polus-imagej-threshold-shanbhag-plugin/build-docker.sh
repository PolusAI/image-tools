#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/imagej-threshold-shanbhag-plugin:${version}