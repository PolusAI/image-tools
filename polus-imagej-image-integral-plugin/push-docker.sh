#!/bin/bash

version=$(<VERSION)
docker push polusai/imagej-image-integral-plugin:${version}