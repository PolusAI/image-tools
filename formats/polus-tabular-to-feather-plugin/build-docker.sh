#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/tabular-to-feather-plugin:${version}