#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/feature-extraction-plugin:${version}