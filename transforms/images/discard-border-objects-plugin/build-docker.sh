#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/discard-border-objects-plugin:${version}