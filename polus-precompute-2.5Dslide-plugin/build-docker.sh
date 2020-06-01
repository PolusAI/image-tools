#!/bin/bash

version=$(<VERSION)
docker build . -t mmvihani/polus-precompute-25slide-plugin:${version}
