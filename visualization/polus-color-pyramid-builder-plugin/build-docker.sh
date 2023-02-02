#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/color-pyramid-builder-plugin:${version}