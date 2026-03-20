#!/bin/bash

version=$(<VERSION)
docker build . -t polusai/remove-border-objects-plugin:${version}