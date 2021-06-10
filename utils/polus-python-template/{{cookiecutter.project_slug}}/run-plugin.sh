#!/bin/bash

version=$(<VERSION)
datapath=$(readlink --canonicalize ../data)

# Inputs
{% for inp,val in cookiecutter._inputs.items() -%}
{% if val.type == 'collection' -%}
{{ inp }}=/data/path_to_files
{% else -%}
{{ inp }}=""
{% endif -%}
{% endfor %}
# Output paths
{% for inp,val in cookiecutter._outputs.items() -%}
{{ inp }}=/data/path_to_output
{% endfor %}
docker run --mount type=bind,source=${datapath},target=/data/ \
            --user $(id -u):$(id -g) \
            labshare/{{ cookiecutter.project_slug }}:${version} \
            {% for inp,val in cookiecutter._inputs.items() -%}
            --{{ inp }} $({{ inp }}) {% if not loop.last %}\{% endif %}
            {% endfor -%}
            {% for out,val in cookiecutter._outputs.items() -%}
            --{{ out }} $({{ out }}) {% if not loop.last %}\{% endif %}
            {% endfor -%}