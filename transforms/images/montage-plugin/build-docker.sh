#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/montage-plugin:${version}