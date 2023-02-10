#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/tiledtiff-converter-plugin:${version}
