#!/bin/bash

version=$(<VERSION)
docker push polusai/imagej-threshold-yen-plugin:${version}
