#!/bin/bash
version=$(<VERSION)
docker build . -t polusai/binary-operations-plugin:${version}