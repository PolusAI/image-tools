#!/bin/bash

version=$(<VERSION)
docker push polusai/{{ cookiecutter.project_slug }}:${version}