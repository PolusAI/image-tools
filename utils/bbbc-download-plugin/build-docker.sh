#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/bbbc-download-plugin:${version}
