#!/bin/bash

version=$(<VERSION)
docker push polusai/imagej-sobel-plugin:${version}