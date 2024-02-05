#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/arrow-to-tabular-plugin:${version}
