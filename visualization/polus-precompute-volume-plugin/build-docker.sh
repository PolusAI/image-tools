#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/precompute-volume-plugin:${version}
