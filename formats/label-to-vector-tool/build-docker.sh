#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/label-to-vector-tool:"${version}"
