#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/imagej-copy-iterableinterval-plugin:${version}