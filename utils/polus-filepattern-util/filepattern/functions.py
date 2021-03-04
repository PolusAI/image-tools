import re, copy, typing, logging
import pathlib

# regular expression used to parse the vector information
STITCH_REGEX = r"file: (.*); corr: (.*); position: \((.*), (.*)\); grid: \((.*), (.*)\);"
""" Stitching regular expression """

STITCH_VARIABLES = ['file','correlation','posX','posY','gridX','gridY'] # image stitching values
""" Stitch variables from :attr:`STITCH_REGEX` """

# permitted filepattern variables
VARIABLES = 'rtczyxp'
"""Permitted filepattern variables

Note:
    Thie static variable may change in the next major revision, where the number
    and name of variables can be arbitrary.
"""

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger('filepattern')

def get_regex(pattern: str) -> typing.Tuple[str,str]:
    """ Parse a filename pattern into a regular expression

    The filename pattern used here mimics that used by MIST, where variables and
    positions are encoded into the string. For example, file_c000.ome.tif that
    indicates channel using the _c, the filename pattern would be
    ``file_c{ccc}.ome.tif``. The only possible variables that can be passed into
    the filename pattern are p, x, y, z, c, t, and r. In the case of p, x, and
    y, both x&y must be specified or p must be specified, but if all three are
    specified then an error is thrown.

    If no filepattern is provided, then a universal expression is returned.

    Args:
        pattern: Filename pattern

    Returns:
        Regex used to parse filenames, Variables found in the filename pattern
    """
    logger.debug(f'get_regex: pattern = {pattern}')

    # Initialize the regular expression, escape parentheses if they exist
    regex = pattern.replace('(','\(').replace(')','\)')

    # If no regex was supplied, return universal matching regex
    if pattern == None or pattern == '' :
        return '.*', []

    # Parse variables
    expr = []
    variables = ''
    for g in re.finditer("{{[{}]+\+?}}".format(VARIABLES),pattern):
        expr.append(g.group(0))
        variables += expr[-1][1]
    logger.debug(f'get_regex: variables = {variables}')

    # Validate variable choices
    _val_variables(variables)

    # Generate the regular expression pattern
    for e in expr:
        if e[2] == '+':
            regex = regex.replace(e,"([0-9]+)")
        else:
            regex = regex.replace(e,"([0-9]{"+str(len(e)-2)+"})")
    logger.debug(f'get_regex: regex = {regex}')

    return regex, variables

def output_name(pattern: str,
                files: typing.List[str],
                ind: dict) -> typing.Union[str,None]:
    """ Returns an output name for a single file resulting from multiple images

    This function returns a file output name for the image volume based on the
    name of multiple files used to generate it. All variables are kept the same
    as in the original filename, but variables in the file name pattern that are
    not present in ind are transformed into a range surrounded by ``()``. For
    example, if the following files are processed:

    .. code-block:: bash
    
        image_c000_z000.ome.tif
        image_c000_z001.ome.tif
        image_c000_z002.ome.tif
        image_c001_z000.ome.tif
        image_c001_z001.ome.tif
        image_c001_z002.ome.tif

    then if ``ind = {'c': 0}``, the output filename will be:

    .. code-block:: bash
        
        image_c000_z(000-002).ome.tif

    Args:
        fpattern: A filename pattern indicating variables in filenames
        files: A list  of file names
        ind: A dictionary containing the indices for the file name (e.g.
            ``{'r':1,'t':1}``)

    Returns:
        An output file name
    """

    # Determine the variables that shouldn't change in the filename pattern
    STATICS = [key for key in ind.keys()]

    # If no pattern was supplied, return None
    if pattern==None or pattern=="":
        return None

    for key in ind.keys():
        assert key in VARIABLES, "Input dictionary key not a valid variable: {}".format(key)

    # Parse variables
    expr = []
    variables = []
    for g in re.finditer("{{[{}]+\+?}}".format(VARIABLES),pattern):
        expr.append(g.group(0))
        variables.append(expr[-1][1])

    # Generate the output filename
    fname = pattern
    for e,v in zip(expr,variables):
        if v not in STATICS:
            minval = min([int(b) for i in files for a,b in i.items() if a==v])
            maxval = max([int(b) for i in files for a,b in i.items() if a==v])
            maxlength = max([len(str(b)) for i in files for a,b in i.items() if a==v])
            fname = fname.replace(e,'(' + str(minval).zfill(maxlength) +
                                    '-' + str(maxval).zfill(maxlength) + ')')
        elif v not in ind.keys():
            fname = fname.replace(e,str(0).zfill(len(e)-2))
        else:
            fname = fname.replace(e,str(ind[v]).zfill(len(e)-2))

    return fname

def parse_filename(file_name: str,
                   regex: str,
                   variables: str) -> typing.Union[dict,None]:
    """ Get the variable indices from a file name

    Extract the variable values from a file name. Return as a dictionary.

    For example, if a file name and file pattern are:

    .. code-block: bash
    
        file_x000_y000_c000.ome.tif
        file_x{xxx}_y{yyy}_c{ccc}.ome.tif

    This function will return:
    .. code-block: bash
    
        {'x': 0,
         'y': 0,
         'c': 0}

    Args:
        file_name: List of values parsed from a filename using a filepattern
        regex: A regular expression used to parse the filename.
        variables: String of variables associated with the regex.

    Returns:
        A dictionary if the filename matches the pattern/regex, otherwise None.
    """

    logger.debug(f'parse_filename: file_name = {file_name}')
    logger.debug(f'parse_filename: regex = {regex}')
    logger.debug(f'parse_filename: variables = {variables}')

    '''Initialize loop and output variables'''
    # Get variable values from the filename
    groups = re.match(regex,file_name)
    if groups == None:  # Don't return anything if the filename doesn't match the regex
        logger.debug(f'parse_filename: output = None')
        return None

    r = {}  # Initialize the output

    # Generate the output
    for v in variables:
        r[v] = int(groups.groups()[[ind for ind,i in zip(range(0,len(variables)),variables) if i==v][0]])

    logger.debug(f'parse_filename: output = {r}')

    return r

def parse_vector_line(vector_line: str,
                      regex: str,
                      variables: str) -> typing.Union[dict,None]:
    """ Get variables from one line of a stitching vector file

    This function parses a single line from a stitching vector. It uses
    parse_filename to extract variable values from the file name. It returns a
    dictionary similar to what is returned by parse_filename, except it includes
    stitching variables in the dictionary.

    Args:
        vector_line: A single line from a stitching vector
        regex: A regular expression used to parse the filename.
        variables: String of variables associated with the regex.

    Returns:
        A dict of stitching and filename variables if the line and filename
            match the pattern. None otherwise.
    """

    logger.debug(f'parse_vector_line: vector_line = {vector_line}')
    logger.debug(f'parse_vector_line: regex = {regex}')
    logger.debug(f'parse_vector_line: variables = {variables}')

    ''' Parse the stitching vector line and filename '''
    # parse the information from the stitching vector line
    stitch_groups = list(re.match(STITCH_REGEX,vector_line).groups())
    stitch_info = {key:value for key,value in zip(STITCH_VARIABLES,stitch_groups)}

    # parse the filename (this does all the sanity checks as well)
    r = parse_filename(stitch_info['file'],regex,variables)
    if r == None:
        logger.debug(f'parse_vector_line: output = None')
        return None
    r.update(stitch_info)

    logger.debug(f'parse_vector_line: output = {r}')

    return r

def parse_directory(file_path: typing.Union[str,pathlib.Path],
                    pattern: typing.Optional[str] = None,
                    regex: typing.Optional[str] = None,
                    variables: typing.Optional[str] = None,
                    var_order: typing.Optional[str] = "rtczyx") -> typing.Tuple[dict,dict]:
    """ Parse files in a directory

    This function extracts the variables value  from each filename in a
    directory and places them in a dictionary that allows retrieval using
    variable values. For example, if there is a folder with filenames using the
    pattern ``file_x{xxx}_y{yyy}_c{ccc}.ome.tif``, then the output will be a
    dictionary with the following structure:

    .. code-block:: bash
        
        output_dictionary[r][t][c][z][y][x]

    To access the filename with values ``x=2``, ``y=3``, and ``c=1``:
    
    .. code-block:: bash
    
        output_dictionary[-1][-1][1][-1][3][2]

    The -1 values are placeholders for variables that were undefined by the
    pattern. The value stored in the deepest layer of the dictionary is a list
    of all files that match the variable values. For a well formed filename
    pattern, the length of the list at each set of coordinates should be one,
    but there are some use cases which makes it beneficial to store many
    filenames at each set of coordinates (see below).

    A custom variable order can be returned using the var_order keyword
    argument. When set, this changes the structure of the output dictionary.
    Using the previous example, if the var_order value was set to ``xyc``, then
    to access the filename matching ``x=2``, ``y=3``, and ``c=1``:

    .. code-block:: bash
    
        output_dictionary[2][3][1]

    The variables in var_order do not need to match the variables in the
    pattern, but this will cause overloaded lists to be returned. Again using
    the same example as before, if the var_order was set to ``xy``, then
    accessing the file associated with ``x=2`` and ``y=3`` will return a list of
    all filenames that match ``x=2`` and ``y=3``, but each filename will have a
    different ``c`` value. This may be useful in applications where filenames
    want to be grouped by a particular attribute (channel, replicate, etc).

    Note:
        The uvals return value is a list of unique values for each variable
        index, but not all combinations of variables are valid in the
        dictionary. It is possible that one level of the dictionary has
        different child values.

    Args:
        file_path: path to a folder containing files to parse
        pattern: A file name pattern. If ``pattern`` is not defined, then both
            ``variables`` and ``var_order`` must be defined.
        variables: String of variables associated with the regex.
        var_order: A string indicating the order of variables in a nested
            output dictionary

    Returns:
        A file index dictionary, a dictionary of unique values for each variable
    """

    if isinstance(file_path,str):
        file_path = pathlib.Path(file_path)

    file_path = file_path.resolve(strict=True)

    files = file_path.iterdir()

    return _parse(parse_filename,
                  files,
                  pattern,
                  regex,
                  variables,
                  var_order)

def parse_vector(file_path: str,
                 pattern: typing.Optional[str] = None,
                 regex: typing.Optional[str] = None,
                 variables: typing.Optional[str] = None,
                 var_order: typing.Optional[str] = "rtczyx") -> typing.Tuple[dict,dict]:
    """ Parse files in a stitching vector

    This function works exactly as parse_directory, except it parses files in a
    stitching vector. In addition to the variable values contained in the file
    dictionary returned by this function, the values associated with the file
    are also contained in the dictionary.

    The format for a line in the stitching vector is as follows:
    ``file: (filename); corr: (correlation)); position: (posX, posY); grid: (gridX, gridY);``

    posX and posY are the pixel positions of an image within a larger stitched
    image, and gridX and gridY are the grid positions for each image.

    Note:
        A key difference between this function and parse_directory is the
        value stored under the ``file`` key. This function returns only the name
        of an image parsed from the stitching vector, while the value returned
        by parse_dictionary is a full path to an image.

    Args:
        file_path: path to a stitching vector file
        pattern: A file name pattern. If ``pattern`` is not defined, then both
            ``variables`` and ``var_order`` must be defined.
        variables: String of variables associated with the regex.
        var_order: A string indicating the order of variables in a nested
            output dictionary

    Returns:
        A file index dictionary, a dictionary of unique values for each variable
    """

    # Build the output dictionary
    with open(file_path,'r') as fr:
        files = [f for f in fr]

    return _parse(parse_vector_line,
                  files,
                  pattern,
                  regex,
                  variables,
                  var_order,
                  False)

def get_matching(files: dict,
                 var_order: str,
                 out_var: dict = None,
                 **kwargs):
    """ Get filenames that have defined variable values

    This gets all filenames that match a set of variable values. Variables must
    be one of :attr:`VARIABLES`, and the inputs must be uppercase. The following
    example code would return all files that have c=0:

    .. code::

        pattern = "file_x{xxx}_y{yyy}_c{ccc}.ome.tif"
        file_path = "./path/to/files"
        files = parse_directory(file_path,pattern,var_order="cyx")
        channel_zero = get_matching(files,"cyx",C=0)

    Multiple coordinates can be used simultaneously, so in addition to C=0 in
    the above example, it is also possible to include Y=0. Further, each
    variable can be a list of values, and the returned output will contain
    filenames matching any of the input values.

    Args:
        files: A file dictionary (see :func:`parse_directory`)
        var_order: A string indicating the order of variables in a nested
            output dictionary
        out_var: Variable to store results, used for recursion
        kwargs: One of :attr:`VARIABLES`, must be uppercase, can be single value
            or a list of values

    Returns:
        A list of file dictionaries matching the input values
    """
    # Initialize the output variable if needed
    if out_var == None:
        out_var = []

    # If there is no var_order, then files should be a list of files.
    if len(var_order)==0:
        if not isinstance(files,list):
            TypeError('Expected files to be a list since var_order is empty.')
        out_var.extend(files)
        return

    for arg in kwargs.keys():
        assert arg==arg.upper() and arg.lower() in VARIABLES, \
            f"Input keyword arguments must be uppercase variables (one of {VARIABLES})"

    if var_order[0].upper() in kwargs.keys():
        if isinstance(kwargs[var_order[0].upper()],list): # If input was already a list
            v_iter = kwargs[var_order[0].upper()]
        else:                                             # If input was not a list, make it a list
            v_iter = [kwargs[var_order[0].upper()]]
    else:
        v_iter = [i for i in files.keys()]
    v_iter.sort()

    for v_i in v_iter:
        if v_i not in files.keys():
            continue
        get_matching(files[v_i],var_order[1:],out_var,**kwargs)
    return out_var

def infer_pattern(files: typing.Union[typing.List[pathlib.Path],typing.List[str]]) -> str:
    """Guess a filepattern given a list of files

    This method attempts to guess a ``filepattern`` that would apply to a list
    of files. There is no gauruntee of a match.
    
    This method will only work if the non-numeric characters in a list of file
    names are identical. For example, this function will work on the following
    list of files:
    
    .. code-block:: bash
    
        Experiment 1 x01 y01 c01.ome.tif
        Experiment 1 x01 y01 c02.ome.tif
        Experiment 1 x01 y01 c03.ome.tif
        Experiment 1 x01 y02 c01.ome.tif
        Experiment 1 x01 y02 c02.ome.tif
        Experiment 1 x01 y02 c03.ome.tif
        
    The following list of files would not work, because the non-numeric
    characters are different
    
    .. code-block:: bash
    
        Experiment 1 x01 y01 DAPI.ome.tif
        Experiment 1 x01 y01 FITC.ome.tif
        Experiment 1 x01 y01 TXRD.ome.tif
        Experiment 1 x01 y02 DAPI.ome.tif
        Experiment 1 x01 y02 FITC.ome.tif
        Experiment 1 x01 y02 TXRD.ome.tif

    Args:
        files: a list of files or pathlib.Path objects

    Returns:
        str: A ``filepattern`` that matches the list of files
    """
    
    files_str = [file if isinstance(file,str) else file.name for file in files]
    
    if len(files_str) == 0:
        return ''
    
    pattern = files_str[0]
    regex, _ = get_regex(pattern)
    
    logger.debug('starting pattern: {}'.format(pattern))
    
    for file in files_str:
        
        if re.match(regex,file) == None:
            
            pattern = sw_search(pattern,file)
            regex, _ = get_regex(pattern)
            
            logger.debug('file: {}'.format(file))
            logger.debug('new pattern: {}'.format(pattern))
            
    logger.debug('final pattern: {}'.format(pattern))
            
    return pattern

def sw_search(pattern: str,
              filename: str) -> str:
    """ Use a Smith-Waterman algorithm variant to modify a filepattern
    
    This function uses a modified
    `Smith-Waterman <https://en.wikipedia.org/wiki/Smith%E2%80%93Waterman_algorithm>`_
    to update the input pattern so that it will operate on the filename. This
    algorithm assumes that only numeric values might change and all text in the
    filename should be identical.
    
    In general, this function should not be directly called. Instead, consider
    using the :any:`infer_pattern` function.
    
    Note:
        There is no gauruntee that the variable definitions in the input pattern
        will be in the same order or position of the pattern returned by this
        function.

    Args:
        pattern: A filepattern
        filename: A filename to compare against the pattern

    Returns:
        A new filepattern that matches the filename
    """
    
    '''Decompose the pattern'''
    numbers = "0123456789<>"

    # Split the old pattern apart, where splits occur at variable definitions
    vlist = [(0,0,"","")]
    for v in VARIABLES:
        
        matches = re.search("{{[{}]+\+?}}".format(v),pattern)
        
        if matches != None:
            if "+" in matches.group(0):
                pattern = pattern.replace(matches.group(0),">",1)
            else:
                pattern = pattern.replace(matches.group(0),"<" * (len(matches.group(0))-2),1)
        else:
            break
        
    logger.debug(f'sw_search: pattern = {pattern}')
    logger.debug(f'sw_search: filename = {filename}')
    
    '''Create the scoring function'''
    sab = {
        'numeric': {
            'match': 2,
            'penalty': 1
        },
        # Larger bonus and penalty for matching/mismatching non-numerics
        'alpha': {
            'match': 5,
            'penalty': 3
        }
    }
    
    '''Creat the scoring matrix'''
    m = len(pattern)
    n = len(filename)
    matrix = [[0]*(n+1) for _ in range(m+1)]
    
    '''Populate the scoring matrix'''
    for pi, p in enumerate(pattern):
        
        p_is_numeric = p in numbers
        
        for fi, f in enumerate(filename):
            
            f_is_numeric = f in numbers
            
            scores = [0]
            
            # generate the similarity score
            if f_is_numeric:
                if p==f or p in "<>":
                    s = matrix[pi][fi] + sab['numeric']['match']
                else:
                    s = matrix[pi][fi] - sab['numeric']['match']
            else:
                if p==f:
                    s = matrix[pi][fi] + sab['alpha']['match']
                else:
                    s = matrix[pi][fi] - sab['alpha']['match']
            scores.append(s)
            
            # calculate the gap scores
            if f_is_numeric:
                if p_is_numeric:
                    wi = matrix[pi+1][fi] - sab['numeric']['penalty']
                    wj = matrix[pi][fi+1] - sab['numeric']['penalty']
                else:
                    wi = matrix[pi+1][fi] - sab['alpha']['penalty']
                    wj = matrix[pi][fi+1] - sab['alpha']['penalty']
            else:
                wi = matrix[pi+1][fi] - sab['alpha']['penalty']
                wj = matrix[pi][fi+1] - sab['alpha']['penalty']
            
            scores.extend([wi,wj])
            
            # Assign the score
            matrix[pi+1][fi+1] = max(scores)
    
    '''Traceback'''
    # Find the best score
    best_score = matrix[m][n]
    row = m
    col = n
    for r in range(1,m+1):
        for c in range(1,n+1):
            if matrix[r][c] > best_score:
                best_score = matrix[r][c]
                row = r
                col = c
                
    # Traceback, building a new pattern template
    pattern_template = filename[col-1]
    last_row = row
    last_col = col
    while True:     # Loop until we reach best_score == 0
        
        # Default to the next set of characters
        r = row - 1
        c = col - 1
        best_score = matrix[r][c]
        
        if matrix[row][col - 1] > best_score:
            best_score = matrix[row][col-1]
            r = row
            c = col - 1
            
        if matrix[row - 1][col] > best_score:
            best_score = matrix[row-1][col]
            r = row - 1
            c = col
            
        if best_score == 0:
            break
        
        row = r
        col = c
        
        # Default to the matching value if available
        if filename[col-1] == pattern[row-1] and last_col != col and last_row != row:
            pattern_template = filename[col-1] + pattern_template
        else:
            
            # If the values don't match, throw and error if they are nonnumeric
            if filename[col-1] not in numbers or pattern[row-1] not in numbers:
                raise ValueError('There are nonnumeric characters that do not match.')
            
            # If we have progressed a row and column, add a static placeholder
            if last_col != col and last_row != row:
                if pattern[row-1] == '>':
                    pattern_template = '>' + pattern_template
                else:
                    pattern_template = '<' + pattern_template
                
            # We didn't progress on either a row or column.
            # This means the lengths don't match, so add a variable placeholder
            else:
                pattern_template = '>' + pattern_template
            
        last_row = row
        last_col = col
    
    '''Construct a new filepattern'''
    pattern = pattern_template
    vi = 0
    for match in re.finditer('[<>]+',pattern_template):
        if ">" in match.group(0):
            vdef = '{{{}+}}'.format(VARIABLES[vi])
        else:
            vdef = '{{{}}}'.format(VARIABLES[vi]*len(match.group(0)))
        pattern = pattern.replace(match.group(0),vdef,1)
        vi += 1
    
    return pattern

def _parse(parse_function: typing.Callable,
           files: typing.List[str],
           pattern: typing.Optional[str] = None,
           regex: typing.Optional[str] = None,
           variables: typing.Optional[str] = None,
           var_order: typing.Optional[str] = 'rtczyx',
           sort: bool = True) -> typing.Tuple[dict,dict]:

    ''' Validate Inputs '''
    logger.debug(f'_parse: parse_function = {parse_function.__name__}()')

    regex, variables = _validate_pat_reg_var(pattern,regex,variables)
    logger.debug(f'_parse: regex = {regex}')
    logger.debug(f'_parse: variables = {variables}')

    # validate variables in var_order
    _val_variables(var_order)
    logger.debug(f'_parse: var_order = {var_order}')

    '''Initialize loop and output variables'''
    # initialize the output
    if variables != "":
        file_ind = {}                     # file dictionary
    else:
        file_ind = []
    uvals = {key:[] for key in var_order} # unique values

    '''Build the file index dictionary'''
    FILE_VARIABLES = {key:-1 for key in var_order}

    # Build the output dictionary
    for f in files:

        # Parse filename values
        if isinstance(f,pathlib.Path):
            fv = parse_function(f.name,regex,variables)
        else:
            fv = parse_function(f,regex,variables)

        # If the filename doesn't match the pattern, don't include it
        if fv == None:
            continue

        # Fill in missing values
        file_variables = copy.deepcopy(FILE_VARIABLES)
        file_variables.update(fv)

        # Generate the layered dictionary using the specified ordering
        temp_dict = file_ind
        for key in var_order:

            # Create a new key,value pair if it's missing
            if file_variables[key] not in temp_dict.keys():

                # Create a new key for the layer if it doesn't exist
                if file_variables[key] not in uvals[key]:
                    uvals[key].append(file_variables[key])

                # If not the last variable, create a new dictionary
                if var_order[-1] != key:
                    temp_dict[file_variables[key]] = {}

                # If the last variable, create a list to hold data
                else:
                    temp_dict[file_variables[key]] = []

            # Grab the reference for the next layer in the dictionary
            temp_dict = temp_dict[file_variables[key]]

        # Add the file information at the deepest layer
        new_entry = {}
        new_entry['file'] = f
        if file_variables != None:
            for key, value in file_variables.items():
                new_entry[key] = value

        temp_dict.append(new_entry)

    if sort:
        for key in uvals.keys():
            uvals[key].sort()

    return file_ind, uvals

def _validate_pat_reg_var(pattern,regex,variables):
    """ Validates pattern, regex, variable inputs """

    # If regex and variables, just validate the variables
    if regex != None and variables != None:
        _val_variables(variables)

    # If pattern is defined, get the corersponding regex and variables
    elif pattern != None:
        regex, variables = get_regex(pattern)

    # Either pattern or regex&variables must be defined
    else:
        raise ValueError(f"""Either pattern must be provided as a keyword argument, or both regex and variables must be provided as input keyword arguments. Received the following keyword arguments:
    {{
        'regex': {regex},
        'variables': {variables},
        'pattern': {pattern}
    }}""")

    return regex,variables

def _val_variables(variables: str) -> None:
    """ Validate file pattern variables

    Variables for a file pattern should only contain the values in
    :attr:`VARIABLES`. In addition to this, only a linear positioning variable
    (p) or an x,y positioning variable should be present, but not both.

    There is no return value for this function. It throws an error if an invalid
    variable is present.

    Args:
        variables: a string of variables, e.g. 'rtxy'
    """

    for v in variables:
        assert v in VARIABLES, f"File pattern variables must be one of {VARIABLES}"