#!/bin/bash

version=$(<VERSION)
docker push polusai/imagej-threshold-maxentropy-plugin:${version}