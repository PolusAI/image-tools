#!/bin/bash

version=$(<VERSION)
docker push polusai/imagej-threshold-percentile-plugin:${version}