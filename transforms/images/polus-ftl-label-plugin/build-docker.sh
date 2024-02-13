#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/ftl-label-plugin:"${version}"