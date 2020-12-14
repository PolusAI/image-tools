#!/bin/bash

version=$(<VERSION)
docker build . -t labshare/polus-image-assembler-plugin:${version}