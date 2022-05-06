#!/bin/bash

version=$(<VERSION)
docker push polusai/imagej-threshold-apply-plugin:${version}