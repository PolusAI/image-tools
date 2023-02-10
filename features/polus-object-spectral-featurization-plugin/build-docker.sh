#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/object-spectral-featurization-plugin:${version}