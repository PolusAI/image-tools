#!/bin/bash

version=$(<VERSION)
docker build . -t mmvihani/polus-precompute-mesh-plugin:${version}
