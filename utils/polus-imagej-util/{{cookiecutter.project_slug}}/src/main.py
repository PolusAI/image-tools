import typing, os, argparse, logging
import ij_converter
import jpype, imagej, scyjava
import numpy as np
import filepattern
from pathlib import Path
from bfio.bfio import BioReader, BioWriter

"""
This file was automatically generated from an ImageJ plugin generation pipeline.
"""

{% if cookiecutter.scalability == 'threshold' %}

def create_iterable(path, ij):
    
    # Read the image with BioReader
    with BioReader(path) as br:
        
        # Create the numpy array of the image
        numpy_image = np.squeeze(br[:, :, 0:1, 0, 0])
        fname = path.name
        metadata = br.metadata
        
        # Close the BioReader
        br.close()
        
    # Convert the image to a IterableInterval object
    iterable_interval = ij_converter.to_java(
        ij, 
        numpy_image, 
        'IterableInterval'
        )
    
    return iterable_interval, fname, metadata


def create_histogram(path, ij):
    
    # Create the iterable interval object from the tile
    iterable_interval, fname, metadata = create_iterable(path, ij)
    
    # Define the histogram mapper class
    Mapper = jpype.JClass('net.imglib2.histogram.Real1dBinMapper')
    
    # Instantiate a mapper
    mapper = Mapper(
        jpype.JDouble(0),       # min bin value
        jpype.JDouble(2**16),   # max bin value
        jpype.JLong(1024),      # number of bins
        jpype.JBoolean(False)   # track values outside of range
        )
    
    # Define a histogram class
    Histogram = jpype.JClass('net.imglib2.histogram.Histogram1d')
    
    # Instantiate a histogram
    histogram = Histogram(iterable_interval, mapper)
    
    return histogram
{% endif -%}
{% if cookiecutter.scalability == 'fft-filter' %}
def pad_image(fp, y, x, padding_size):

    # Get the file path to the image to pad
    img_path = fp.get_matching(Y=y, X=x)[0]['file']
    print(img_path)
    
    print('Reading {} - position x:{} y:{}'.format(img_path.name, x, y))
    
    # Read the image to pad
    br = BioReader(img_path, backend='python')
    meta = br.metadata
    img = br[:]
    br.close()
    original_shape = img.shape
    print(original_shape)
    
    # Pad the image with symmetric reflection
    padded_img = np.pad(img, original_shape, mode='symmetric')

    # Iterate over each row of tiles
    for r in range(-1, 2):
        
        # Define the current y (row) to read
        Y = y + r
        
        # Iterate over each column in the current row
        for c in range(-1, 2):
            
            # Define the current x (column) to read
            X = x + c
            
            # Try to get file path to the current image in the 3x3 grid
            next_image = fp.get_matching(X=X, Y=Y)
            
            if len(next_image) == 0:
                print('No image in position y:{}, x:{}'.format(Y,X))
            
            else:
                print(next_image)
                
                # Read the next image
                br = BioReader(Path(next_image[0]['file']))
                img = br[:]
                br.close()
                
                # Define the current tiles array indices
                y1 = (r + 1)*original_shape[0]
                y2 = y1 + original_shape[0]
                x1 = (c + 1)*original_shape[0]
                x2 = x1 + original_shape[0]
                
                print('y1:', y1)
                print('y2:', y2)
                print('x1:', x1)
                print('x2:', x2)
                
                # Slice in the current image
                padded_img[y1:y2, x1:x2] = img
    
    # Define the final indices of the paddded image
    i1 = original_shape[0] - padding_size
    i2 = original_shape[0]*2 + padding_size
    
    return padded_img[i1:i2, i1:i2], original_shape, meta
{% endif -%}

# Import environment variables
POLUS_LOG = getattr(logging,os.environ.get('POLUS_LOG','INFO'))
POLUS_EXT = os.environ.get('POLUS_EXT','.ome.tif')

# Initialize the logger
logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("main")
logger.setLevel(POLUS_LOG)

def main({#- Required inputs -#}
         {% for inp,val in cookiecutter._inputs.items() -%}
         {%     if val.required and inp != 'out_input' -%}
         {%         if val.type=="boolean" -%}
         _{{ inp }}: bool,
         {%         elif val.type=="collection" -%}
         _{{ inp }}: Path,
         {%         else -%}
         _{{ inp }}: str,
         {%         endif -%}
         {%     endif -%}
         {% endfor -%}
         {%- if cookiecutter.scalability == 'fft-filter' -%}
         _pattern: str,
         {%- endif %}
         {#- Required Outputs (all outputs are required, and should be a Path) -#}
         {% for inp,val in cookiecutter._outputs.items() -%}
         _{{ inp }}: Path,
         {% endfor -%}
         {#- Optional inputs -#}
         {% for inp,val in cookiecutter._inputs.items() -%}
         {%     if not val.required -%}
         {%         if val.type=="boolean" -%}
         _{{ inp }}: typing.Optional[bool] = None,
         {%         elif val.type=="collection" -%}
         _{{ inp }}: typing.Options[Path] = None,
         {%         else -%}
         _{{ inp }}: typing.Optional[str] = None,
         {%         endif -%}
         {%     endif -%}
         {% endfor -%}) -> None:
             
    """ Initialize ImageJ """ 
    # Bioformats throws a debug message, disable the loci debugger to mute it
    def disable_loci_logs():
        DebugTools = scyjava.jimport("loci.common.DebugTools")
        DebugTools.setRootLevel("WARN")
    scyjava.when_jvm_starts(disable_loci_logs)
    
    # This is the version of ImageJ pre-downloaded into the docker container
    logger.info('Starting ImageJ...')
    
    ij = imagej.init(
        "sc.fiji:fiji:2.1.1+net.imagej:imagej-legacy:0.37.4",
        headless=True
        )
    
    logger.info('Loaded ImageJ version: {}'.format(ij.getVersion()))
    
    """ Validate and organize the inputs """
    args = []
    argument_types = []
    arg_len = 0

    {% for inp,val in cookiecutter._inputs.items() if inp != 'out_input' %}
    # Validate {{ inp }}{% if inp != "opName" %}
    {{ inp }}_types = { {% for i,v in val.call_types.items() %}
        "{{ i }}": "{{ v }}",{% endfor %}
    }
    
    # Check that all inputs are specified 
    if _{{ inp }} is None and _opName in list({{ inp }}_types.keys()):
        raise ValueError('{} must be defined to run {}.'.format('{{ inp }}',_opName))
    {%- if val.type == "collection"%}
    elif _{{ inp }} != None:
        {{ inp }}_type = {{ inp }}_types[_opName]
    
        # switch to images folder if present
        if _{{ inp }}.joinpath('images').is_dir():
            _{{ inp }} = _{{ inp }}.joinpath('images').absolute()
        
        # Check that input path is a directory
        if not _{{ inp }}.is_dir():
            raise FileNotFoundError('The {} collection directory does not exist'.format(_{{ inp }}))
        
        # Add the list of images to the arguments (images) list
        # There will be a single list for each collection input within args list
        {%- if cookiecutter.scalability == 'fft-filter' and val.title in ['in1', 'inpDir'] %}
        
        # Instantiate the filepatter object
        {{ inp }}_fp = filepattern.FilePattern(_{{inp}}, _pattern)
        
        # Add the list of images to the arguments (images) list
        # There will be a single list for each collection input within args list
        args.append([f[0] for f in {{ inp }}_fp() if f[0]['file'].is_file()])
        if arg_len == 0:
            arg_len = len(args[-1])
        
        {%- elif cookiecutter.scalability == 'fft-filter' and val.title in ['in2', 'kernel'] %}
        
        # Infer the file pattern of the collection
        pattern_guess = filepattern.infer_pattern(_{{inp}}.iterdir())
        
        # Instantiate the filepatter object
        {{ inp }}_fp = filepattern.FilePattern(_{{inp}}, pattern_guess)
        
        {{ inp }}_path = [f[0]['file'] for f in {{ inp }}_fp() if f[0]['file'].is_file()]
        
        {% else %}
        # Infer the file pattern of the collection
        pattern_guess = filepattern.infer_pattern(_{{inp}}.iterdir())
        
        # Instantiate the filepatter object
        {{ inp }}_fp = filepattern.FilePattern(_{{inp}}, pattern_guess)
        
        # Add the list of images to the arguments (images) list
        # There will be a single list for each collection input within args list
        args.append([f[0]['file'] for f in {{ inp }}_fp() if f[0]['file'].is_file()])
        if arg_len == 0:
            arg_len = len(args[-1])
        {% endif %}
        
    else:
        argument_types.append(None)
        args.append([None])
    {%- else %}
    else:
        {{ inp }} = None
    {%- endif %}
    {% else %}
    {{ inp }}_values = [{% for v in val.options %}
        "{{v}}",{% endfor %}
    ]
    assert _{{ inp }} in {{ inp }}_values, '{{ inp }} must be one of {}'.format({{ inp }}_values)
    {% endif %}{%- endfor %}
    
    # This ensures each input collection has the same number of images
    # If one collection is a single image it will be duplicated to match length
    # of the other input collection - only works when 1 input is a collection
    for i in range(len(args)):
        if len(args[i]) == 1:
            args[i] = args[i] * arg_len
            
    # Define the output data types for each overloading method
    {%- for out,val in cookiecutter._outputs.items() %}
    {{ out }}_types = { {% for i,v in val.call_types.items() %}
        "{{ i }}": "{{ v }}",{%- endfor %}
    }{%- endfor %}
    
    {%- if cookiecutter.scalability == 'independent' %}
    # Attempt to convert inputs to java types and run the pixel indepent op
    try:
        for ind, (
            {%- for inp,val in cookiecutter._inputs.items() -%}
            {%- if val.type=='collection' and inp != 'out_input' %}{{ inp }}_path,{% endif -%}
            {%- endfor %}) in enumerate(zip(*args)):
            
            {%- for inp,val in cookiecutter._inputs.items() if val.type=='collection' and inp != 'out_input' %}
            {%- if val.type=='collection' %}
            if {{ inp }}_path != None:

                # Load the first plane of image in {{ inp }} collection
                logger.info('Processing image: {}'.format({{ inp }}_path))
                {{ inp }}_br = BioReader({{ inp }}_path)
                
                # Convert to appropriate numpy array
                {{ inp }} = ij_converter.to_java(ij, np.squeeze({{ inp }}_br[:,:,0:1,0,0]),{{ inp }}_type)
                {%- if loop.first %}
                metadata = {{ inp }}_br.metadata
                fname = {{ inp }}_path.name
                dtype = ij.py.dtype({{ inp }})
                # Save the shape for out input
                shape = ij.py.dims({{ inp }})
                {%- endif %}
            {%- endif %}{% endfor %}
            
            {%- for inp,val in cookiecutter._inputs.items() if val.type != 'collection' and inp != 'opName' and inp != 'out_input' %}
            if _{{ inp }} is not None:
                {{ inp }} = ij_converter.to_java(ij, _{{ inp }},{{ inp }}_types[_opName],dtype)
            {% endfor %}
            
            # Generate the out input variable if required
            {%- for inp,val in cookiecutter._inputs.items() if inp == 'out_input' %}
            {{ inp }} = ij_converter.to_java(ij, np.zeros(shape=shape, dtype=dtype), 'IterableInterval')
            {% endfor %}
            
            logger.info('Running op...')
            {% for i,v in cookiecutter.plugin_namespace.items() %}
            {%- if loop.first %}if{% else %}elif{% endif %} _opName == "{{ i }}":
                {{ v }}
            {% endfor %}
            logger.info('Completed op!')
            
            {%- for inp,val in cookiecutter._inputs.items() if inp != 'out_input' %}
            {%- if val.type=='collection' %}
            if {{ inp }}_path != None:
                {{ inp }}_br.close()
            {%- endif %}{% endfor %}
            
            {% for out,val in cookiecutter._outputs.items() -%}

            # Saving output file to {{ out }}
            logger.info('Saving...')
            {{ out }}_array = ij_converter.from_java(ij, {{ out }},{{ out }}_types[_opName])
            bw = BioWriter(_{{ out }}.joinpath(fname),metadata=metadata)
            bw.Z = 1
            bw.dtype = {{ out }}_array.dtype
            bw[:] = {{ out }}_array.astype(bw.dtype)
            bw.close()
            {%- endfor %}
            
    except:
        logger.error('There was an error, shutting down jvm before raising...')
        raise
            
    finally:
        # Exit the program
        logger.info('Shutting down jvm...')
        del ij
        jpype.shutdownJVM()
        logger.info('Complete!')
        
    {% elif cookiecutter.scalability == 'threshold' %}
    
    try:
        
        logger.info('Computing threshold value...')
        
        # Create a tile count
        tile_count = 0
        
        for {%- for inp,val in cookiecutter._inputs.items() -%}
            {%- if val.type=='collection' and inp != 'out_input' %} {{ inp }}_path, in zip(*args):
            
            # Check if any tiles have been processed
            if tile_count == 0:
                
                # Create the initial histogram
                histogram = create_histogram({{ inp }}_path, ij)
            
            else:
                
                # Convert the image to an iterable interval
                iterable_interval, fname, metadata = create_iterable({{ inp }}_path, ij)
                
                # Add the image tile to the histogram
                histogram.addData(iterable_interval)
            
            tile_count += 1
            {% endif -%}{%- endfor %}
        
        # Calculate the threshold value
        {{ cookiecutter.compute_threshold }}
        
        # Check if array was returned
        if isinstance(threshold, jpype.JClass('java.util.ArrayList')):
            
            # Get the threshold value, disregard the errMsg output
            threshold = threshold[0]
        
        logger.info('The threshold value is {}'.format(threshold))
        
        for {%- for inp,val in cookiecutter._inputs.items() -%}
        {%- if val.type=='collection' and inp != 'out_input' %} {{ inp }}_path, in zip(*args):
        
                    # Load the first plane of image in {{ inp }} collection
                    logger.info('Processing image: {}'.format({{ inp }}_path))
                    
                    # Convert the image to an iterable interval
                    iterable_interval, fname, metadata = create_iterable({{ inp }}_path, ij)
                    
                    # Apply the threshold
                    out = ij.op().threshold().apply(iterable_interval, threshold)
                    
                    # Write image to file
                    logger.info('Saving image {}'.format(fname))
                    out_array = ij_converter.from_java(ij, out, 'Iterable')
                    bw = BioWriter(_out.joinpath(fname), metadata=metadata)
                    bw.Z = 1
                    bw.dtype = out_array.dtype
                    bw[:] = out_array.astype(bw.dtype)
                    bw.close()     
        {% endif -%}{%- endfor %}
        
    except:
        logger.error('There was an error, shutting down jvm before raising...')
        raise
            
    finally:
        # Exit the program
        logger.info('Shutting down jvm...')
        del ij
        jpype.shutdownJVM()
        logger.info('JVM shutdown complete')
    
    {% elif cookiecutter.scalability == 'fft-filter' %}
    
    # Attempt to convert inputs to java types and run the filter op
    try:
        
        {%- for inp,val in cookiecutter._inputs.items() if val.type != 'collection' and inp != 'opName' and inp != 'out_input' %}
            if _{{ inp }} is not None:
                {{ inp }} = ij_converter.to_java(ij, _{{ inp }},{{ inp }}_types[_opName],dtype)
        {% endfor %}
        
        {%- for inp,val in cookiecutter._inputs.items() if val.title in ['in2', 'kernel'] and val.type=='collection' %}
        # Load the kernel image
        logger.info('Loading image: {}'.format({{ inp }}_path[0]))
        {{ inp }}_br = BioReader({{ inp }}_path[0])
        
        # Convert to appropriate numpy array
        {{ inp }} = ij_converter.to_java(ij, np.squeeze({{ inp }}_br[:,:,0:1,0,0]),{{ inp }}_type)
        {{ inp }}_br.close()
        
        kernel_shape = ij.py.dims({{ inp }})
        {% endfor %}
        
        for ind, (
            {%- for inp,val in cookiecutter._inputs.items() if val.type=='collection' and inp not in ['out_input', 'in2', 'kernel'] -%}
            {{ inp }}_path,
            {%- endfor %}) in enumerate(zip(*args)):
            
            {%- for inp,val in cookiecutter._inputs.items() if val.type=='collection' and inp not in ['out_input', 'in2', 'kernel'] %}
            if {{ inp }}_path != None:
                
                {%- if loop.first %}
                # Load the first plane of image in {{ inp }} collection
                logger.info('Processing image: {}'.format({{ inp }}_path))
                
                # Define x and y spatial position of current tile
                x = {{ inp }}_path['x']
                y = {{ inp }}_path['y']
                
                # Save input collection file name and data type
                fname = {{ inp }}_path['file'].name
                
                # Pad the tile
                padded_img, orginal_shape, metadata = pad_image(
                    fp={{ inp }}_fp, y=y, x=x, padding_size=kernel_shape[0]
                    )
                
                # Convert to appropriate numpy array
                {{ inp }} = ij_converter.to_java(ij, padded_img,{{ inp }}_type)
                
                # Save the shape and data type for out input array
                shape = ij.py.dims({{ inp }})
                dtype = ij.py.dtype({{ inp }})
                {%- endif %}
            {% endfor %}
            
            # Generate the out input variable if required
            {%- for inp,val in cookiecutter._inputs.items() if inp == 'out_input' %}
            {{ inp }} = ij_converter.to_java(ij, np.zeros(shape=shape, dtype=dtype), 'IterableInterval')
            {% endfor %}
            
            logger.info('Running op...')
            {% for i,v in cookiecutter.plugin_namespace.items() %}
            {%- if loop.first %}if{% else %}elif{% endif %} _opName == "{{ i }}":
                {{ v }}
            {% endfor %}
            logger.info('Completed op!')
            
            {% for out,val in cookiecutter._outputs.items() -%}
            # Saving output file to {{ out }}
            logger.info('Saving...')
            {{ out }}_array = ij_converter.from_java(ij, {{ out }},{{ out }}_types[_opName])
            
            
            # Define padding indices to trim
            i1 = kernel_shape[0]
            i2 = orginal_shape[0] + kernel_shape[0]
            
            {{ out }}_array = {{ out }}_array[i1:i2, i1:i2]
            
            bw = BioWriter(_{{ out }}.joinpath(fname),metadata=metadata)
            bw.Z = 1
            bw.dtype = {{ out }}_array.dtype
            bw[:] = {{ out }}_array.astype(bw.dtype)
            bw.close()
            {%- endfor %}
            
    except:
        logger.error('There was an error, shutting down jvm before raising...')
        raise
            
    finally:
        # Exit the program
        logger.info('Shutting down jvm...')
        del ij
        jpype.shutdownJVM()
        logger.info('Complete!')
    
    {% else %}
    # If plugin scale type is not defined
    logger.info('Plugin scale type not developed, shutting down jvm without running op...')
    del ij
    jpype.shutdownJVM()
    {% endif %}

if __name__=="__main__":

    # Setup Command Line Arguments
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='{{ cookiecutter.project_short_description }}')
    
    # Add command-line argument for each of the input arguments
    {% for inp,val in cookiecutter._inputs.items() if inp != 'out_input' -%}
    parser.add_argument('--{{ val.title }}', dest='{{ inp }}', type=str,
                        help='{{ val.description }}', required={{ val.required }})
    {% endfor %}
    # Add command-line argument for each of the output arguments
    {%- for out,val in cookiecutter._outputs.items() %}
    parser.add_argument('--{{ val.title }}', dest='{{ out }}', type=str,
                        help='{{ val.description }}', required=True)
    {% endfor %}
    {%- if cookiecutter.scalability == 'fft-filter' %}
    parser.add_argument('--pattern', dest='pattern', type=str,
                        help='Input collection file pattern', required=True)
    {% endif %}
    """ Parse the arguments """
    args = parser.parse_args()
    
    # Input Args
    {%- for inp,val in cookiecutter._inputs.items() if inp != 'out_input' %}
    {% if val.type=="boolean" -%}
    _{{ inp }} = args.{{ inp }} == 'true'
    {% elif val.type=="collection" -%}
    _{{ inp }} = Path(args.{{ inp }})
    {% else -%}
    _{{ inp }} = args.{{ inp }}
    {% endif -%}
    logger.info('{{ val.title }} = {}'.format(_{{ inp }}))
    {% endfor %}
    {%- if cookiecutter.scalability == 'fft-filter' %}
    _pattern = args.pattern
    logger.info('pattern = {}'.format(_pattern))
    {% endif %}
    # Output Args
    {%- for out,val in cookiecutter._outputs.items() %}
    _{{ out }} = Path(args.{{ out }})
    logger.info('{{ val.title }} = {}'.format(_{{ out }}))
    {%- endfor %}
    
    main(
    {%- filter indent(5) %}
    {%- for inp,val in cookiecutter._inputs.items() if inp != 'out_input' -%}
    _{{ inp }}=_{{ inp }},
    {% endfor -%}
    {%- if cookiecutter.scalability == 'fft-filter' %}
    _pattern = _pattern,
    {% endif -%}
    {%- for inp,val in cookiecutter._outputs.items() -%}
    _{{ inp }}=_{{ inp }}{% if not loop.last %},{% endif %}{% endfor %}{% endfilter -%}
    )