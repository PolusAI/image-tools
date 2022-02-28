#!/bin/bash

version=$(<VERSION)
docker push polusai/imagej-threshold-huang-plugin:${version}