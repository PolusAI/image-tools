#!/bin/bash

version=$(<VERSION)
docker push polusai/polus-imagej-threshold-maxentropy-plugin:${version}