#!/bin/bash
version=$(<VERSION)
docker build . -t polusai/basic-flatfield-correction-plugin:${version}
