#!/bin/bash

version=$(<VERSION)
docker push polusai/imagej-copy-iterableinterval-plugin:${version}