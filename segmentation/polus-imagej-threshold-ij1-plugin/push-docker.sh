#!/bin/bash

version=$(<VERSION)
docker push polusai/imagej-threshold-ij1-plugin:${version}