import re, copy
from pathlib import Path

STITCH_VARS = ['file','correlation','posX','posY','gridX','gridY'] # image stitching values
VARIABLES = 'rtczyxp'

def val_variables(variables):
    """ Validate file pattern variables
    
    Variables for a file pattern should only contain the values in filepattern.VARIABLES.
    In addition to this, only a linear positioning variable (p) or an x,y positioning
    variable should be present, but not both.
    There is no return value for this function. It throws an error if an invalid variable
    is present.
    Inputs:
        variables - a string of variables, e.g. 'rtxy'
    Outputs:
        None
    """

    for v in variables:
        assert v in VARIABLES, "File patter variables must be one of {}".format(VARIABLES)

    if 'p' in variables:
        assert 'x' not in variables and 'y' not in variables, "Either x and/or y may be defined or p may be defined, but not both."

def get_regex(pattern):
    """ Parse a filename pattern into a regular expression
    
    The filename pattern used here mimics that used by MIST, where variables and
    positions are encoded into the string. For example, file_c000.ome.tif that
    indicates channel using the _c, the filename pattern would be file_c{ccc}.ome.tif.
    The only possible variables that can be passed into the filename pattern are
    p, x, y, z, c, t, and r. In the case of p, x, and y, both x&y must be specified
    or p must be specified, but if all three are specified then an error is thrown.
    If no filepattern is provided, then a universal expression is returned.
    Inputs:
        pattern - Filename pattern
    Outputs:
        regex - Regex used to parse filenames
        variables - Variables found in the filename pattern
    """

    # Initialize the regular expression
    regex = pattern

    # If no regex was supplied, return universal matching regex
    if pattern==None or pattern=='':
        return '.*', []
    
    # Parse variables
    expr = []
    variables = []
    for g in re.finditer("{{[{}]+}}".format(VARIABLES),pattern):
        expr.append(g.group(0))
        variables.append(expr[-1][1])
        
    # Validate variable choices
    val_variables(variables)
    
    # Generate the regular expression pattern
    for e in expr:
        regex = regex.replace(e,"([0-9]{"+str(len(e)-2)+"})")
        
    return regex, variables

def output_name(pattern,files,ind):
    """ Returns an output name for a single file resulting from multiple images
    This function returns a file output name for the image volume
    based on the name of multiple files used to generate it.
    All variables are kept the same as in the original filename,
    but variables in the file name pattern that are not present in ind
    are transformed into a range surrounded by <>.
    For example, if the following files are processed:
    
    image_c000_z000.ome.tif
    image_c000_z001.ome.tif
    image_c000_z002.ome.tif
    image_c001_z000.ome.tif
    image_c001_z001.ome.tif
    image_c001_z002.ome.tif
    
    then if ind = {'c': 0}, the output filename will be:
    image_c000_z<000-002>.ome.tif
    Inputs:
        fpattern - A filename pattern indicating variables in filenames
        files - A parsed dictionary of file names
        ind - A dictionary containing the indices for the file name (i.e. {'r':1,'t':1})
    Outputs:
        fname - an output file name
    """

    # Determine the variables that shouldn't change in the filename pattern
    STATICS = [key for key in ind.keys()]

    # If no pattern was supplied, return default image name
    if pattern==None or pattern=='':
        return 'image.ome.tif'
    
    for key in ind.keys():
        assert key in VARIABLES, "Input dictionary key not a valid variable: {}".format(key)
    
    # Parse variables
    expr = []
    variables = []
    for g in re.finditer("{{[{}]+}}".format(VARIABLES),pattern):
        expr.append(g.group(0))
        variables.append(expr[-1][1])
        
    # Generate the output filename
    fname = pattern
    for e,v in zip(expr,variables):
        if v not in STATICS:
            minval = min([int(i) for i in files.keys()])
            maxval = max([int(i) for i in files.keys()])
            fname = fname.replace(e,'<' + str(minval).zfill(len(e)-2) +
                                    '-' + str(maxval).zfill(len(e)-2) + '>')
        elif v not in ind.keys():
            fname = fname.replace(e,str(0).zfill(len(e)-2))
        else:
            fname = fname.replace(e,str(ind[v]).zfill(len(e)-2))
        
    return fname

def parse_filename(file_name,pattern=None,regex=None,variables=None,return_empty=True):
    """ Get the x, y, p, z, c, t, and r indices from a file name
    
    Extract the variable values from a file name. Return as a dictionary.
    For example, if a file name and file pattern are:
    file_x000_y000_c000.ome.tif
    file_x{xxx}_y{yyy}_c{ccc}.ome.tif
    This function will return:
    {
        'x': 0,
        'y': 0,
        'c': 0
    }
    Inputs:
        file_name - List of values parsed from a filename using a filename pattern
        pattern - A file name pattern. Either this or regex must be defined (not both).
        regex - A regular expression used to parse the filename.
        return_empty - Returns undefined variables as -1
    Outputs:
        index - The value of the dimension
    """
    # Get the regex if not defined, and validate inputs
    if pattern != None:
        regex,variables = get_regex(pattern)
    elif regex == None:
        ValueError('Either pattern or regex must be specified.')
    elif variables == None:
        ValueError('If regex is an input, then variables must be an input.')
    else:
        val_variables(variables)

    # Get variable values from the filename
    groups = re.match(regex,file_name)
    if groups == None:  # Don't return anything if the filename doesn't match the regex
        return None

    r = {}  # Initialize the output

    # Initialize variable iterator, include undefined variables
    iter_vars = VARIABLES
    if 'p' in variables:
        iter_vars = iter_vars.replace('x','')
        iter_vars = iter_vars.replace('y','')
    else:
        iter_vars = iter_vars.replace('p','')

    # Generate the output
    for v in iter_vars:
        if v not in variables:
            if return_empty:
                r[v] = -1
        else:
            r[v] = int(groups.groups()[[ind for ind,i in zip(range(0,len(variables)),variables) if i==v][0]])

    return r

def parse_vector_line(vector_line,pattern=None,regex=None,variables=None,return_empty=True):
    """ Get the file, corr, posX, posY, gridX, and gridY information from a vector
    
    This function parses a single line from a stitching vector. It uses parse_filename
    to extract variable values from the file name. It returns a dictionary similar
    to what is returned by parse_filename, except it includes stitching variables
    in the dictionary.
    Inputs:
        vector_line - A single line from a stitching vector
        pattern - A file name pattern. Either this or regex must be defined (not both).
        regex - A regular expression used to parse the filename.
        return_empty - Returns undefined variables as -1
    Outputs:
        index - The value of the dimension
    """

    # regular expression used to parse the vector information
    line_regex = r"file: (.*); corr: (.*); position: \((.*), (.*)\); grid: \((.*), (.*)\);"
    
    # parse the information from the stitching vector line
    stitch_groups = list(re.match(line_regex,vector_line).groups())
    stitch_info = {key:value for key,value in zip(STITCH_VARS,stitch_groups)}
    
    # parse the filename (this does all the sanity checks as well)
    r = parse_filename(stitch_info['file'],pattern,regex,variables,return_empty)
    if r == None:
        return None
    r.update(stitch_info)

    return r

def parse_directory(file_path,pattern,var_order='rtczyx'):
    """ Parse files in a directory
    
    This function extracts the variables in from each filename in a directory and places
    them in a dictionary that allow retrieval using variable values. For example, if there
    is a folder with filenames using the pattern file_x{xxx}_y{yyy}_c{ccc}.ome.tif, then
    the output will be a dictionary with the following structure:
    output_dictionary[r][t][c][z][y][x]
    To access the filename with values x=2, y=3, and c=1:
    output_dictionary[-1][-1][1][-1][3][2]
    The -1 values are placeholders for variables that were undefined by the pattern. The value
    stored in the deepest layer of the dictionary is a list of all files that match the variable
    values. For a well formed filename pattern, the length of the list at each set of
    coordinates should be one, but there are some use cases which makes it beneficial to
    store many filenames at each set of coordinates (see below).
    A custom variable order can be returned using the var_order keyword argument. When set,
    this changes the structure of the output dictionary. Using the previous example,
    if the var_order value was set to `xyc`, then to access the filename matching x=2, y=3,
    and c=1:
    output_dictionary[2][3][1]
    The variables in var_order do not need to match the variables in the pattern, but this
    will cause overloaded lists to be returned. Again using the same example as before,
    if the var_order was set to 'xy', then accessing the file associated with x=2 and y=3
    will return a list of all filenames that match x=2 and y=3, but each filename will have
    a different c value. This may be useful in applications where filenames want to be grouped
    by a particular attribute (channel, replicate, etc).
    NOTE: The uvals return value is a list of unique values for each variable index, but not
    all combinations of variables are valid in the dictionary. It is possible that one level
    of the dictionary has different child values.
    Inputs:
        file_path - path to a folder containing files to parse
        pattern - A file name pattern.
        var_order - A string indicating the order of variables in a nested output dictionary
    Outputs:
        file_ind - The output dictionary containing all files matching the file pattern, sorted
                   by variable value
        uvals - Unique variables for each 
    """

    # validate the variable order
    val_variables(var_order)

    # get regular expression from file pattern
    regex, variables = get_regex(pattern)

    # initialize the output
    if len(variables) == 0:
        file_ind = []
    else:
        file_ind = {}
    files = [f.name for f in Path(file_path).iterdir() if f.is_file()]
    files.sort()

    # Unique values for each variable
    uvals = {key:[] for key in var_order}

    # Build the output dictionary
    for f in files:
        
        # Parse filename values
        variables = parse_filename(f,pattern)

        # If the filename doesn't match the pattern, don't include it
        if variables == None:
            continue
        
        # Generate the layered dictionary using the specified ordering
        temp_dict = file_ind
        if isinstance(file_ind,dict):
            for key in var_order:
                if variables[key] not in temp_dict.keys():
                    if variables[key] not in uvals[key]:
                        uvals[key].append(variables[key])
                    if var_order[-1] != key:
                        temp_dict[variables[key]] = {}
                    else:
                        temp_dict[variables[key]] = []
                temp_dict = temp_dict[variables[key]]
        
        # Add the file information at the deepest layer
        new_entry = {}
        new_entry['file'] = str(Path(file_path).joinpath(f).absolute())
        if variables != None:
            for key, value in variables.items():
                new_entry[key] = value
        temp_dict.append(new_entry)

    for key in uvals.keys():
        uvals[key].sort()
    
    return file_ind, uvals

def parse_vector(file_path,pattern,var_order='rtczyx'):
    """ Parse files in a stitching vector
    
    This function works exactly as parse_directory, except it parses files in a stitching
    vector. In addition to the variable values contained in the file dictionary returned
    by this function, the values associated with the file are also contained in the
    dictionary.
    
    The format for a line in the stitching vector is as follows:
    file: (filename); corr: (correlation)); position: (posX, posY); grid: (gridX, gridY);
    
    posX and posY are the pixel positions of an image within a larger stitched image, and
    gridX and gridY are the grid positions for each image.
    
    NOTE: A key difference between this function and parse_directory is the value stored
          under the 'file' key. This function returns only the name of an image parsed
          from the stitching vector, while the value returned by parse_dictionary is a
          full path to an image.
    Inputs:
        file_path - path to a folder containing files to parse
        pattern - A file name pattern.
        var_order - A string indicating the order of variables in a nested output dictionary
    Outputs:
        file_ind - The output dictionary containing all files matching the file pattern, sorted
                   by variable value
        uvals - Unique variables for each 
    """

    # validate the variable order
    val_variables(var_order)

    # get regular expression from file pattern
    regex, variables = get_regex(pattern)

    # initialize the output
    if len(variables) == 0:
        file_ind = []
    else:
        file_ind = {}

    # Unique values for each variable
    uvals = {key:[] for key in var_order}

    # Build the output dictionary
    with open(file_path,'r') as fr:
        for f in fr:
            
            # Parse filename values
            variables = parse_vector_line(f,pattern)

            # If the filename doesn't match the patter, don't include it
            if variables == None:
                continue
            
            # Generate the layered dictionary using the specified ordering
            temp_dict = file_ind
            if isinstance(file_ind,dict):
                for key in var_order:
                    if variables[key] not in temp_dict.keys():
                        if variables[key] not in uvals[key]:
                            uvals[key].append(variables[key])
                        if var_order[-1] != key:
                            temp_dict[variables[key]] = {}
                        else:
                            temp_dict[variables[key]] = []
                    temp_dict = temp_dict[variables[key]]
            
            # Add the file information at the deepest layer
            temp_dict.append(variables)

    for key in uvals.keys():
        uvals[key].sort()
    
    return file_ind, uvals

def get_matching(files,var_order,out_var=None,**kwargs):
    """ Get filenames that have defined variable values
    
    This gets all filenames that match a set of variable values. Variables must be one of
    filename.VARIABLES, and the inputs must be uppercase. The following example code would
    return all files that have c=0:
    pattern = "file_x{xxx}_y{yyy}_c{ccc}.ome.tif"
    file_path = "./path/to/files"
    files = parse_directory(file_path,pattern,var_order='cyx')
    channel_zero = get_matching(files,'cyx',C=0)
    Multiple coordinates can be used simultaneously, so in addition to C=0 in the above example,
    it is also possible to include Y=0. Further, each variable can be a list of values, and the
    returned output will contain filenames matching any of the input values.
    Inputs:
        files - A file dictionary (see parse_directory)
        var_order - A string indicating the order of variables in a nested output dictionary
        out_var - Variable to store results, used for recursion
        kwargs - One of filepatter.VARIABLES, must be uppercase, can be single values or a list of values
    Outputs:
        out_var - A list of all files matching the input values
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
        assert arg==arg.upper() and arg.lower() in VARIABLES, "Input keyword arguments must be uppercase variables (one of R, T, C, Z, Y, X, P)"
    
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

class FilePattern():
    """ Main class for handling filename patterns
    
    Most of the functions in filepattern.py return complicated variable structures that might
    be difficult to use in an abstract way. This class provides tools to use the above functions
    more simple. In particular, the iterate function is an iterable that permits simple
    iteration over filenames with specific values and grouped by any desired variable.
    """
    var_order = 'rtczyx'
    files = {}
    uniques = {}

    def __init__(self,file_path,pattern,var_order=None):
        self.pattern, self.variables = get_regex(pattern)
        self.path = file_path

        if var_order:
            val_variables(var_order)
            self.var_order = var_order

        self.files, self.uniques = parse_directory(file_path,pattern,var_order=self.var_order)

    # Get filenames matching values for specified variables
    def get_matching(self,**kwargs):
        """ Get all filenames matching specific values
    
        This function runs the get_matching function using the objects file dictionary.
        Inputs:
            kwargs - One of filepatter.VARIABLES, must be uppercase, can be single values or a list of values
        Outputs:
            files - A list of all files matching the input values
        """
        # get matching files
        files = get_matching(self.files,self.var_order,out_var=None,**kwargs)
        return files

    def iterate(self,group_by=[],**kwargs):
        """ Iterate through filenames
    
        This function is an iterable. On each call, it returns a list of filenames that matches a set of
        variable values. It iterates through every combination of variable values.
        Variables designated in the group_by input argument are grouped together. So, if group_by='zc', 
        then each iteration will return all filenames that have constant values for each variable except z
        and c.
        In addition to the group_by variable, specific variable arguments can also be included as with the
        get_matching function.
        Inputs:
            group_by - String of variables by which the output filenames will be grouped
            kwargs - One of filepatter.VARIABLES, must be uppercase, can be single values or a list of values
        Outputs:
            iter_files - A list of all files matching the input values
        """
        # If self.files is a list, no parsing took place so just loop through the files
        if isinstance(self.files,list):
            for f in self.files:
                yield f
            return

        # Generate the values to iterate through
        iter_vars = {}
        for v in self.var_order:
            if v in group_by:
                continue
            elif v.upper() in kwargs.keys():
                if isinstance(kwargs[v.upper()],list):
                    iter_vars[v] = copy.deepcopy(kwargs[v.upper()])
                else:
                    iter_vars[v] = [kwargs[v.upper()]]
            else:
                iter_vars[v] = copy.deepcopy(self.uniques[v])
        
        # Find the shallowest variable in the dictionary structure
        shallowest = None
        for v in iter_vars.keys():
            if -1 in iter_vars[v] and len(iter_vars[v]):
                continue
            else:
                shallowest = v
                break

        # If shallowest is undefined, return all file names
        if shallowest == None:
            yield get_matching(self.files,self.var_order,**{key.upper():iter_vars[key][0] for key in iter_vars.keys()})
            return

        # Loop through every combination of files
        while len(iter_vars[shallowest])>0:
            # Get list of filenames and return as iterator
            iter_files = []
            iter_files = get_matching(self.files,self.var_order,**{key.upper():iter_vars[key][0] for key in iter_vars.keys()})
            if len(iter_files)>0:
                yield iter_files

            # Delete last iteration indices
            for v in reversed(self.var_order):
                if v in group_by:
                    continue
                del iter_vars[v][0]
                if len(iter_vars[v])>0:
                    break
                elif v == shallowest:
                    break
                iter_vars[v] = copy.deepcopy(self.uniques[v])
                
class VectorPattern(FilePattern):
    """ Main class for handling stitching vectors
    
    This class works nearly identically to FilePattern, except it works with lines
    inside of a stitching vector. As with FilePattern, the iterate method will iterate
    through values, which in the case of VectorPattern are parsed lines of a stitching
    vector.
    """
    
    var_order = 'rtczyx'
    files = {}
    uniques = {}
    
    def __init__(self,file_path,pattern,var_order=None):
        self.pattern, self.variables = get_regex(pattern)
        self.path = file_path

        if var_order:
            val_variables(var_order)
            self.var_order = var_order

        self.files, self.uniques = parse_vector(file_path,pattern,var_order=self.var_order)