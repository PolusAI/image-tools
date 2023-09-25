#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/region-segmentation-comparison-plugin:${version}
