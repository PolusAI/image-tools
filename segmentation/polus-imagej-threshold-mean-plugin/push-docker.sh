#!/bin/bash

version=$(<VERSION)
docker push polusai/imagej-threshold-mean-plugin:${version}