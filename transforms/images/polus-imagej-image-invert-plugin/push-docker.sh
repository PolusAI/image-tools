#!/bin/bash

version=$(<VERSION)
docker push polusai/polus-imagej-image-invert-plugin:${version}