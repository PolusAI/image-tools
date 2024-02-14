#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/region-segmentation-eval:${version}
