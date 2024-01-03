#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/image-dimension-stacking-plugin:${version}
