#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/rxiv-download-tool:${version}
