#!/bin/bash

version=$(<VERSION)
docker push labshare/polus-smp-training-plugin:"${version}"
