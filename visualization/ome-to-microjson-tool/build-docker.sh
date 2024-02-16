#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/ome-to-microjson-tool:${version}
