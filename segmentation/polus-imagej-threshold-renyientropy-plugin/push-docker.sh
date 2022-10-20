#!/bin/bash

version=$(<VERSION)
docker push polusai/imagej-threshold-renyientropy-plugin:${version}