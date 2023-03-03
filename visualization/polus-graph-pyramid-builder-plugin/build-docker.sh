#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/graph-pyramid-builder-plugin:${version}
