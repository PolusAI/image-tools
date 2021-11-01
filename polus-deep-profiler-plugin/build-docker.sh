#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/polus-deep-profiler-plugin:${version}