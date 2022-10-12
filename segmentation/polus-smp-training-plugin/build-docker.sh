#!/bin/bash

rm -rf src/utils/__pycache__

version=$(<VERSION)
docker build . -t polusai/smp-training-plugin:"${version}"
