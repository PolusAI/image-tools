#!/bin/bash

version=$(<VERSION)
docker build . -t labshare/polus-feature-extraction-3d-plugin:${version}
