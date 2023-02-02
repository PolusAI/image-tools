#!/bin/bash

version=$(<VERSION)
docker build . -f label_to_vector.Dockerfile -t polusai/label-to-vector-plugin:"${version}"
docker build . -f vector_to_label.Dockerfile -t polusai/vector-to-label-plugin:"${version}"
