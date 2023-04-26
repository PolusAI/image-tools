#!/bin/bash
version=$(<VERSION)
docker build . -t polusai/recycle-vector-plugin:${version}