{% if cookiecutter.use_bfio == "True" -%}
from bfio.bfio import BioReader, BioWriter
{%- endif %}
{% if cookiecutter.use_filepattern == "True" -%}
import filepattern
{%- endif %}
import argparse, logging
import numpy as np
import typing, os
from pathlib import Path

# Import environment variables
POLUS_LOG = getattr(logging,os.environ.get('POLUS_LOG','INFO'))
POLUS_EXT = os.environ.get('POLUS_EXT','.ome.tif')

# Initialize the logger
logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("main")
logger.setLevel(POLUS_LOG)

{# Define indentation levels -#}
{% set level2 = 4 if cookiecutter.use_bfio == "True" else 0 -%}

def awesome_function(input_data: np.ndarray
                     ) -> np.ndarray:
    """Awesome function (actually just a template)
    
    This function should do something, but for now just returns the input.

    Args:
        input_data: A numpy array.
        
    Returns:
        np.ndarray: Returns the input image.
    """
    
    return input_data
    
{# This generates the main function definition -#}
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
    pattern = filePattern if filePattern is not None else '.*'
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
    
    logger.info(f'Processing image: {file["file"]}')
    
    # Load the input image
    logger.debug(f'Initializing BioReader for {file["file"]}')
    with BioReader(file['file']) as br:
        
        input_extension = ''.join([s for s in file['file'].suffixes[-2:] if len(s) < 5])
        out_name = file['file'].name.replace(input_extension,POLUS_EXT)
        out_path = {{ cookiecutter._outputs.keys()|first }}.joinpath(out_name)
        
        # Initialize the output image
        logger.debug(f'Initializing BioReader for {out_path}')
        with BioWriter(out_path,metadata=br.metadata) as bw:
            
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
    {% elif val.type=="collection" -%}
    {{ inp }} = Path(args.{{ inp }})
    if ({{ inp }}.joinpath('images').is_dir()):
        # switch to images folder if present
        {{ inp }} = {{ inp }}.joinpath('images').absolute()
    {% else -%}
    {{ inp }} = args.{{ inp }}
    {% endif -%}
    logger.info('{{ inp }} = {}'.format({{ inp }}))
    {% endfor %}
    {%- for out,val in cookiecutter._outputs.items() -%}
    {{ out }} = Path(args.{{ out }})
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