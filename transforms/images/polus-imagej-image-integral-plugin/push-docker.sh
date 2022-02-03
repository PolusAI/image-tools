#!/bin/bash

version=$(<VERSION)
docker push polusai/polus-imagej-image-integral-plugin:${version}