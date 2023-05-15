#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/feather-to-tabular-plugin:${version}