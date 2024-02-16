#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/montage-tool:"${version}"
