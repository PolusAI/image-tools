#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/omero-download-tool:${version}
