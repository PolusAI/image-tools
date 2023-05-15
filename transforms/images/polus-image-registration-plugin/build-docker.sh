#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/image-registration-plugin:${version}