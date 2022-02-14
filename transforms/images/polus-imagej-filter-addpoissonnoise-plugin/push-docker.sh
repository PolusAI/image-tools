#!/bin/bash

version=$(<VERSION)
docker push polusai/polus-imagej-filter-addpoissonnoise-plugin:${version}