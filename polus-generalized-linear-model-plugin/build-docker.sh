#!/bin/bash

version=$(<VERSION)
docker build . -t labshare/polus-generalized-linear-model-plugin:${version}