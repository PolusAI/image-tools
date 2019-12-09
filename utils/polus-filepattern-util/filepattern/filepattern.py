import re, copy
from pathlib import Path

VARIABLES = 'rtczyxp'

def val_variables(variables):
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
    # Get the regex is not defined
    if pattern != None:
        regex,variables = get_regex(pattern)
    elif regex == None:
        ValueError('Either pattern or regex must be specified.')
    elif variables == None:
        ValueError('If regex is an input, then variables must be an input.')

    groups = re.match(regex,file_name)
    if groups == None:
        return None

    r = {}

    iter_vars = VARIABLES
    if 'p' in variables:
        iter_vars = iter_vars.replace('x','')
        iter_vars = iter_vars.replace('y','')
    else:
        iter_vars = iter_vars.replace('p','')

    for v in iter_vars:
        if v not in variables:
            if return_empty:
                r[v] = -1
        else:
            r[v] = int(groups.groups()[[ind for ind,i in zip(range(0,len(variables)),variables) if i==v][0]])

    return r

def parse_directory(file_path,pattern,var_order='rtczyx'):
    
    # validate the variable order
    val_variables(var_order)

    # initialize the output
    file_ind = {}
    files = [f.name for f in Path(file_path).iterdir() if f.is_file()]

    # Unique values for each variable
    uvals = {key:[] for key in var_order}

    # Build the output dictionary
    for f in files:
        
        # Parse filename values
        variables = parse_filename(f,pattern)
        if variables == None:
            continue
        
        # Generate the layered dictionary using the specified ordering
        temp_dict = file_ind
        for key in var_order:
            if variables[key] not in temp_dict.keys():
                if variables[key] not in uvals[key]:
                    uvals[key].append(variables[key])
                if var_order[-1] != key:
                    temp_dict[variables[key]] = {}
                else:
                    temp_dict[variables[key]] = []
            temp_dict = temp_dict[variables[key]]
        
        # At the file information at the deepest layer
        new_entry = {}
        new_entry['file'] = str(Path(file_path).joinpath(f).absolute())
        for key, value in variables.items():
            new_entry[key] = value
        temp_dict.append(new_entry)
            
    return file_ind, uvals

def get_matching(files,var_order,out_var=None,**kwargs):
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
    var_order = 'rtczyx'
    files = {}
    uniques = {}

    def __init__(self,file_path,pattern,var_order=None):
        self.pattern = get_regex(pattern)

        if var_order:
            self.var_order = var_order

        self.files, self.uniques = parse_directory(file_path,pattern,var_order=self.var_order)

    # Get filenames matching values for specified variables
    def get_matching(self,**kwargs):
        # get matching files
        files = get_matching(self.files,self.var_order,out_var=None,**kwargs)
        return files

    def iterate(self,group_by=[],**kwargs):
        
        # Generate the values to iterate through
        iter_vars = {}
        for v in self.var_order:
            if v in group_by:
                continue
            elif v.upper() in kwargs.keys():
                iter_vars[v] = kwargs[v]
            else:
                iter_vars[v] = copy.deepcopy(self.uniques[v])
        
        # iterate over the values until the most superficial values are empty
        for v in self.var_order:
            if v not in group_by:
                shallowest = v
                break
        while len(iter_vars[shallowest])>0:
            # Get list of filenames and return as iterator
            iter_files = []
            iter_files = get_matching(self.files,self.var_order,**{key.upper():iter_vars[key][0] for key in iter_vars.keys()})
            if len(iter_files)>0:
                yield iter_files

            # Delete last iteration indices
            for v in reversed(self.var_order):
                if v in group_by or v.upper() in kwargs.keys():
                    continue
                del iter_vars[v][0]
                if len(iter_vars[v])>0:
                    break
                elif v == shallowest:
                    break
                iter_vars[v] = copy.deepcopy(self.uniques[v])