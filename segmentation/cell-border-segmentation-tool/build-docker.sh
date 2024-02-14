#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/cell-border-segmentation-tool:${version}
