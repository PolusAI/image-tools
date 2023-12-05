#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/rxiv-download-plugin:${version}