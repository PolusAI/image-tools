#!/bin/bash

version=$(<VERSION)

docker build . -t polusai/bleed-through-estimation-plugin:"${version}"
