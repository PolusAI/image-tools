#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/feature-evaluation-plugin:${version}
