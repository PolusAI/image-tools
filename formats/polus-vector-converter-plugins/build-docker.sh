#!/bin/bash

version=$(<VERSION)
docker build . -f label_to_vector.Dockerfile -t labshare/polus-label-to-vector-plugin:"${version}"
docker build . -f vector_to_label.Dockerfile -t labshare/polus-vector-label-plugin:"${version}"
