#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/autocropping-plugin:"${version}"
