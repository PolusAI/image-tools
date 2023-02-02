#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/image-assembler-plugin:${version}