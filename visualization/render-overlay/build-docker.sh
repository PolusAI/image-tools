#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/render-overlay-plugin:${version}
