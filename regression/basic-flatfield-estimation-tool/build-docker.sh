#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/basic-flatfield-estimation-tool:"${version}"
