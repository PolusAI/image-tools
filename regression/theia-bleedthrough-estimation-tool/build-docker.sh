#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/theia-bleedthrough-estimation-tool:"${version}"
