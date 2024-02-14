#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/arrow-to-tabular-tool:${version}
