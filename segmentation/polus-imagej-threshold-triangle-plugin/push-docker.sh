#!/bin/bash

version=$(<VERSION)
docker push polusai/imagej-threshold-triangle-plugin:${version}