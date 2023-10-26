#!/bin/bash

version=$(<VERSION)

docker build . -t polusai/roi-relabel-plugin:"${version}"
