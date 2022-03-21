#!/bin/bash

version=$(<VERSION)
docker push polusai/imagej-threshold-shanbhag-plugin:${version}