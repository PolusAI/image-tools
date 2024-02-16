#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/feature-segmentation-eval-tool:${version}
