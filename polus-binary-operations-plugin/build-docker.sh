#!/bin/bash

version=$(<VERSION)
docker build . -t mmvihani/polus-binary-operations-plugin:${version}
