#!/bin/bash

version=$(<VERSION)
docker push polusai/imagej-threshold-isodata-plugin:${version}