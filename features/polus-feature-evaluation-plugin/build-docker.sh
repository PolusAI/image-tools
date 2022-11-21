#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/feature-eval-plugin:${version}
