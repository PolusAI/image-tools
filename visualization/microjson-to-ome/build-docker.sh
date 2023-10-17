#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/microjson-to-ome-plugin:${version}
