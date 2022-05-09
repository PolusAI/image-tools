#!/bin/bash

version=$(<VERSION)

docker build . -t polusai/pytorch-inference-plugin:"${version}"
