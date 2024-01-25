#!/bin/bash

version=$(<VERSION)
docker build . -t {{cookiecutter.container_id}}:${version}
