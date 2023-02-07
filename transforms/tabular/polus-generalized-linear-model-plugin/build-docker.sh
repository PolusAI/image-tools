#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/generalized-linear-model-plugin:${version}