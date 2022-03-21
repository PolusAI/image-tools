#!/bin/bash

version=$(<VERSION)
docker push polusai/polus-imagej-threshold-renyientropy-plugin:${version}