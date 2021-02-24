#!/bin/bash

version=$(<VERSION)
datapath="{{ '../../../data'|abspath }}"

# Inputs
{% for inp,val in cookiecutter._inputs.items() -%}
{{ inp }}="/data/path_to_files"
{% endfor %}
# Output paths
{% for inp,val in cookiecutter._outputs.items() -%}
{{ inp }}="{{ val.type }}"
{% endfor %}
docker run --mount type=bind,source=${datapath},target=/data/ \
            labshare/{{ cookiecutter.project_slug }}:${version} \
            {% for inp,val in cookiecutter._inputs.items() -%}
            --{{ inp }} $({{ inp }}) {% if not loop.last %}\{% endif %}
            {% endfor -%}
            {% for out,val in cookiecutter._outputs.items() -%}
            --{{ out }} $({{ out }}) {% if not loop.last %}\{% endif %}
            {% endfor -%}