#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/cellular-eval-plugin:${version}