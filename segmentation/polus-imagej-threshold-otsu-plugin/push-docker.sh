#!/bin/bash

version=$(<VERSION)
docker push polusai/imagej-threshold-otsu-plugin:${version}