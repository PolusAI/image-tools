#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/imagej-macro-plugin:${version}