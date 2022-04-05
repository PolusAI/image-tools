#!/bin/bash

version=$(<VERSION)
docker push polusai/imagej-image-invert-plugin:${version}