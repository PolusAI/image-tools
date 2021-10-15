#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/polus-image-quality-plugin:${version}