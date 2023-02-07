#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/stack-z-slice-plugin:${version}