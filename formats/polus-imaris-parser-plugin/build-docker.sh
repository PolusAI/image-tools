#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/imaris-parser-plugin:${version}