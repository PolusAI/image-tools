#!/bin/bash

version=$(<VERSION)
docker push polusai/imagej-threshold-li-plugin:${version}