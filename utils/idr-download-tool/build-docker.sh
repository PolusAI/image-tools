#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/idr-download-tool:${version}
