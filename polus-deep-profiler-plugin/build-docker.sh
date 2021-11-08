#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/deep-profiler-plugin:${version}