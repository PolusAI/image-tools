#!/bin/bash

version=$(<VERSION)
docker push labshare/polus-label-to-vector-plugin:"${version}"
docker push labshare/polus-vector-label-plugin:"${version}"
