#!/bin/bash

rm -rf src/utils/__pycache__

version=$(<VERSION)
docker build . -t labshare/polus-smp-training-plugin:"${version}"
