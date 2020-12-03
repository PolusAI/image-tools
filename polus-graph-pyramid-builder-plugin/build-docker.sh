#!/bin/bash

version=$(<VERSION)
docker build . -t labshare/polus-graph-pyramid-builder-plugin:${version}
