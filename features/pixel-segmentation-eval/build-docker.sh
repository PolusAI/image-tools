#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/pixel-segmentation-eval:${version}
