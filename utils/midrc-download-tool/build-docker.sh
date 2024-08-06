#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/midrc-download-tool:${version}
