#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/kaggle-nuclei-segmentation-tool:${version}
