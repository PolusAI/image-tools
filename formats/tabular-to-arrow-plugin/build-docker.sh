#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/tabular-to-arrow-plugin:${version}
