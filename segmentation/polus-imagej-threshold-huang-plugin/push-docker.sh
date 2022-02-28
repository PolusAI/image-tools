#!/bin/bash

version=$(<VERSION)
docker push polusai/polus-imagej-threshold-huang-plugin:${version}