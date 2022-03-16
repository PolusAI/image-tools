#!/bin/bash

version=$(<VERSION)
docker push polusai/polus-imagej-threshold-percentile-plugin:${version}