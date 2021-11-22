#!/bin/bash

version=$(<VERSION)
datapath=$(readlink --canonicalize ../data)

# Inputs
{% for inp,val in cookiecutter._inputs.items() -%}
{% if val.type == 'collection' -%}
{{ inp }}=/data/input
{% else -%}
{{ inp }}=
{% endif -%}
{% endfor %}
# Output paths
{% for inp,val in cookiecutter._outputs.items() -%}
{{ inp }}=/data/output
{% endfor %}
docker run --mount type=bind,source=${datapath},target=/data/ \
            polusai/{{ cookiecutter.project_slug }}:${version} \
            {% for inp,val in cookiecutter._inputs.items() -%}
            --{{ inp }} {% raw %}${{% endraw %}{{ inp }}{% raw %}}{% endraw %} \
            {% endfor -%}
            {% for out,val in cookiecutter._outputs.items() -%}
            --{{ out }} {% raw %}${{% endraw %}{{ out }}{% raw %}}{% endraw %}{% if not loop.last %} \{% endif %}
            {% endfor -%}