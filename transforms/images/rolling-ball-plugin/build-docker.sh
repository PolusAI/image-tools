#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/rolling-ball-plugin:"${version}"
