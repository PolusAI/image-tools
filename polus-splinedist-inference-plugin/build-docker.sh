#!/bin/bash

version=$(<VERSION)
docker build . -t labshare/polus-splinedist-inference-plugin:${version}
