#!/bin/bash
version=$(<VERSION)
docker build . -t polusai/mesmer-training-tool:${version}
