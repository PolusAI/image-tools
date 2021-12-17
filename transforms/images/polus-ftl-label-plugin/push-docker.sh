#!/bin/bash

version=$(<VERSION)
docker push labshare/polus-ftl-label-plugin:"${version}"