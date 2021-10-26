#!/bin/bash

version=$(<VERSION)
docker push labshare/polus-autocropping-plugin:"${version}"
