#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/ome-converter-tool:${version}
