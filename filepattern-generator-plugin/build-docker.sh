#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/filepattern-generator-plugin:${version}