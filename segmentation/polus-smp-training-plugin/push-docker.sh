#!/bin/bash

version=$(<VERSION)
docker push polusai/smp-training-plugin:"${version}"
