#!/bin/bash

version=$(<VERSION)
docker build . -t labshare/polus-precompute-mesh-plugin:${version}
