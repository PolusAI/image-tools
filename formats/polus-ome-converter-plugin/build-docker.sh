#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/ome-converter-plugin:${version}