#!/bin/bash

version=$(<VERSION)
docker push polusai/imagej-filter-addpoissonnoise-plugin:${version}