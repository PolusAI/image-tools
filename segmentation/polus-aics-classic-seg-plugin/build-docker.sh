#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/aics-classic-seg-plugin:${version}