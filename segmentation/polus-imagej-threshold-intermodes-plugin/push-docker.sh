#!/bin/bash

version=$(<VERSION)
docker push polusai/imagej-threshold-intermodes-plugin:${version}