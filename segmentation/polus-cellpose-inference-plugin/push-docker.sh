#!/bin/bash

version=$(<VERSION)
docker push labshare/polus-cellpose-inference-plugin:"${version}"
