#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/pixelwise-eval-plugin:${version}