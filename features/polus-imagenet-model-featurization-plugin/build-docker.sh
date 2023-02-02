#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/imagenet-model-featurization-plugin:${version}