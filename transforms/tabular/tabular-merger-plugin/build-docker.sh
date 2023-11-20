#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/tabular-merger-plugin:${version}
