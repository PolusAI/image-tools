#!/bin/bash

version=$(<VERSION)
docker push polusai/csv-statistics-plugin:"${version}"