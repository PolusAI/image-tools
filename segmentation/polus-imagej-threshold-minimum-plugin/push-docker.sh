#!/bin/bash

version=$(<VERSION)
docker push polusai/imagej-threshold-minimum-plugin:${version}