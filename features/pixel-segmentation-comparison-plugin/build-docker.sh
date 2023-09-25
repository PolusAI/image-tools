#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/pixel-segmentation-comparison-plugin:${version}
