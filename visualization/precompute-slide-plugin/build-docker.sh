#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/precompute-slide-plugin:${version}
