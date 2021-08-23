#!/bin/bash

version=$(<VERSION)
datapath=$(readlink --canonicalize ../data)

# Inputs
{% for inp,val in cookiecutter._inputs.items() -%}
{% if val.type == 'collection' -%}
{{ inp }}=/data/path_to_files
{% elif inp == 'filePattern' -%}
{{ inp }}=.+
{% else -%}
{{ inp }}="{{ inp }}"
{% endif -%}
{% endfor %}
# Output paths
{% for inp,val in cookiecutter._outputs.items() -%}
{{ inp }}=/data/path_to_output
{% endfor %}
# GPU configuration for testing GPU usage in the container
GPUS=all

# Log level, must be one of ERROR, CRITICAL, WARNING, INFO, DEBUG
LOGLEVEL=INFO

docker run --mount type=bind,source=${datapath},target=/data/ \
            --user $(id -u):$(id -g) \
            --gpus ${GPUS} \
            --env POLUS_LOG=${LOGLEVEL} \
            labshare/{{ cookiecutter.project_slug }}:${version} \
            {% for inp,val in cookiecutter._inputs.items() -%}
            --{{ inp }} ${%raw%}{{%endraw%}{{ inp }}{%raw%}}{%endraw%} \
            {% endfor -%}
            {% for out,val in cookiecutter._outputs.items() -%}
            --{{ out }} ${%raw%}{{%endraw%}{{ out }}{%raw%}}{%endraw%} {% if not loop.last %}\{% endif %}
            {% endfor -%}