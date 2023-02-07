#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/image-cluster-annotation-plugin:${version}