#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/multichannel-tiff-plugin:${version}