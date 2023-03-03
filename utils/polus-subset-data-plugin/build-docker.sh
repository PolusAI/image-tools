#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/subset-data-plugin:${version}