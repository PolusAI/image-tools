#!/bin/bash

version=$(<VERSION)
docker build . -t labshare/{{ cookiecutter.project_slug }}:${version}