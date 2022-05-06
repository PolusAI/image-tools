#!/bin/bash

version=$(<VERSION)
docker push polusai/imagej-threshold-moments-plugin:${version}