#!/bin/bash

version=$(<VERSION)
docker build . -t labshare/polus-object-spectral-featurization-plugin:${version}