#!/bin/bash

version=$(<VERSION)
docker build . -f setup.Dockerfile -t vector-label-base

echo "Precompiling cupy methods..."
docker run --entrypoint python3 \
         --user "$(id -u)":"$(id -g)" \
         --gpus all \
         --name vector-label-setup \
         vector-label-base -c \
         "import cupy,numpy,dynamics; dynamics.vectors_to_labels(numpy.random.rand(50).reshape(2,5,5).astype(numpy.float32),numpy.ones((1,5,5),dtype=int),1)"

echo "Committing precompiled container..."
docker commit vector-label-setup vector-label-setup:latest

docker build . -f label_to_vector.Dockerfile -t labshare/polus-label-to-vector-plugin:"${version}"
docker build . -f vector_to_label.Dockerfile -t labshare/polus-vector-label-plugin:"${version}"

docker rm vector-label-setup