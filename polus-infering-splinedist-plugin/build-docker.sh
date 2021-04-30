#!/bin/bash

version=$(<VERSION)
docker build . -t mmvihani/polus-inferring-splinedist-plugin:${version}
