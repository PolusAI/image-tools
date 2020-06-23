#!/bin/bash

version=$(<VERSION)
docker build . -t mmvihani/polus-precompute-volume-plugin:${version}
