#!/bin/bash

version=$(<VERSION)
docker build . -t mmvihani/polus-training-splinedist-plugin:${version}
