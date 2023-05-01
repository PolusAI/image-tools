#!/bin/bash
version=$(<VERSION)
docker build . -t polusai/mesmer-training-plugin:${version}
