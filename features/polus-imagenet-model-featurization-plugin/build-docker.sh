#!/bin/bash

version=$(<VERSION)
docker build . -t labshare/polus-imagenet-model-featurization-plugin:${version}