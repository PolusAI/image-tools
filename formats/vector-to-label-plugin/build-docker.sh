#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/vector-to-label-plugin:"${version}"
