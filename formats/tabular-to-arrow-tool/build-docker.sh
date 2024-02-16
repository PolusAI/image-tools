#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/tabular-to-arrow-tool:${version}
