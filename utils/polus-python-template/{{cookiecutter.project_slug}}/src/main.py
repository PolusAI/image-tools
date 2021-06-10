{% if cookiecutter.use_bfio == "True" -%}
from bfio.bfio import BioReader, BioWriter
{%- endif %}
{% if cookiecutter.use_filepattern == "True" -%}
import filepattern
{%- endif %}
import argparse, logging
import numpy as np
import typing
from pathlib import Path

# Initialize the logger
logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("main")
logger.setLevel(logging.INFO)

{# Define indentation levels -#}
{% set level2 = 4 if cookiecutter.use_bfio == "True" else 0 -%}

{#- This generates the main function definition -#}
{#- Required inputs are arguments -#}
{#- Optional inputs are keyword arguments -#}
def main({#- Required inputs -#}
         {% for inp,val in cookiecutter._inputs.items() -%}
         {%     if val.required -%}
         {%         if val.type=="boolean" -%}
         {{ inp }}: bool,
         {%         elif val.type=="collection" and cookiecutter.use_bfio -%}
         {{ inp }}: Path,
         {%         else -%}
         {{ inp }}: str,
         {%         endif -%}
         {%     endif -%}
         {% endfor -%}
         {#- Required Outputs (all outputs are required, and should be a Path) -#}
         {% for inp,val in cookiecutter._outputs.items() -%}
         {{ inp }}: Path,
         {% endfor -%}
         {#- Optional inputs -#}
         {% for inp,val in cookiecutter._inputs.items() -%}
         {%     if not val.required -%}
         {%         if val.type=="boolean" -%}
         {{ inp }}: typing.Optional[bool] = None,
         {%         elif val.type=="collection" and cookiecutter.use_bfio -%}
         {{ inp }}: typing.Options[Path] = None,
         {%         else -%}
         {{ inp }}: typing.Optional[str] = None,
         {%         endif -%}
         {%     endif -%}
         {% endfor -%}) -> None:
    """ Main execution function
    
    All functions in your code must have docstrings attached to them, and the
    docstrings must follow the Google Python Style:
    https://www.sphinx-doc.org/en/master/usage/extensions/example_google.html
    """
    
    {# Initialize a filepattern object if filepattern is going to be used -#}
    {%- if cookiecutter.use_filepattern == "True" %}
    pattern = filepattern.infer_pattern(p.name for p in {% for inp,val in cookiecutter._inputs.items() if val.type=='collection' -%}
        {% if loop.first %}{{ inp }}{% endif %}
        {%- endfor -%}.iterdir())
    fp = filepattern.FilePattern(
        {%- for inp,val in cookiecutter._inputs.items() if val.type=='collection' -%}
        {% if loop.first %}{{ inp }}{% endif %}
        {%- endfor -%},pattern)
    
    for files in fp:
        # get the first file
        file = files.pop()
        
    {%- else %}
    files = list({% for inp,val in cookiecutter._inputs.items() if val.type=='collection' -%}
        {% if loop.first %}{{ inp }}{% endif %}
        {%- endfor -%}.iterdir())
    
    for file in files:
        
    {%- endif %}
    {#- Use bfio if requested #}
    {%- if cookiecutter.use_bfio == "True" %}
    {%- filter indent(level2,True) %}
    
    # Load the input image
    with BioReader(file['file']) as br:
        
        # Initialize the output image
        with BioWriter({{ cookiecutter._outputs.keys()|first }}.joinpath(file['file'].name),metadtata=br.metadata) as bw:
            
            # This is where the magic happens, replace this part with your method
            bw[:] = awesome_function(br[:])
    {%- endfilter %}
    {%- endif %}

if __name__=="__main__":

    ''' Argument parsing '''
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='{{ cookiecutter.project_short_description }}')
    
    # Input arguments
    {% for inp,val in cookiecutter._inputs.items() -%}
    parser.add_argument('--{{ inp }}', dest='{{ inp }}', type=str,
                        help='{{ val.description }}', required={{ val.required }})
    {% endfor -%}
    
    # Output arguments
    {%- for out,val in cookiecutter._outputs.items() %}
    parser.add_argument('--{{ out }}', dest='{{ out }}', type=str,
                        help='{{ val.description }}', required=True)
    {% endfor %}
    # Parse the arguments
    args = parser.parse_args()
    {% for inp,val in cookiecutter._inputs.items() -%}
    {% if val.type=="boolean" -%}
    {{ inp }} = args.{{ inp }} == 'true'
    logger.info('{{ inp }} = {}'.format({{ inp }}))
    {% else -%}
    {{ inp }} = args.{{ inp }}
    {% if val.type=="collection" -%}
    if (Path.is_dir(Path(args.{{ inp }}).joinpath('images'))):
        # switch to images folder if present
        fpath = str(Path(args.{{ inp }}).joinpath('images').absolute())
    {% endif -%}
    logger.info('{{ inp }} = {}'.format({{ inp }}))
    {% endif -%}
    {% endfor %}
    {%- for out,val in cookiecutter._outputs.items() -%}
    {{ out }} = args.{{ out }}
    logger.info('{{ out }} = {}'.format({{ out }}))
    {%- endfor %}
    
    main(
    {%- filter indent(5) %}
    {%- for inp,val in cookiecutter._inputs.items() -%}
    {{ inp }}={{ inp }},
    {% endfor -%}
    {%- for inp,val in cookiecutter._outputs.items() -%}
    {{ inp }}={{ inp }}{% if not loop.last %},{% endif %}{% endfor %}{% endfilter -%}
    )