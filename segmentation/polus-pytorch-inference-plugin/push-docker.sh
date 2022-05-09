#!/bin/bash

version=$(<VERSION)
docker push polusai/pytorch-inference-plugin:"${version}"
