#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/tabular-converter-plugin:${version}
