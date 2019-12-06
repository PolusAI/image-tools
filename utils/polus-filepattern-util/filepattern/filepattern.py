import re

VARIABLES = 'pxyzctr'

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
    for g in re.finditer(r"\{[{pxyzctr}]+\}",pattern):
        expr.append(g.group(0))
        variables.append(expr[-1][1])
        
    # Verify variables are one of pxyzctr
    for v in variables:
        assert v in VARIABLES, "Invalid variable: {}".format(v)
            
    # Verify that either x&y are defined or p is defined, but not both
    if 'x' in variables and 'y' in variables:
        assert 'p' not in variables, "Variable p cannot be defined if x and y are defined."
    elif 'p' in variables:
        assert 'x' not in variables and 'y' not in variables, "Neither x nor y can be defined if p is defined."
    else:
        ValueError("Either p must be defined or x and y must be defined.")
        
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
    for g in re.finditer(r"\{[pxyzctr]+\}",pattern):
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

# def get_values(filename,pattern):
#     """ Parse the x, y, p, z, c, t, or r value from a filename
    
#     When files are parsed, the variables are used in an index to provide
#     a method to reference a specific file name by its dimensions. This
#     function returns the variable index based on the input filename pattern.
#     Inputs:
#         var_list - List of values parsed from a filename using regex
#         variables - List of permitted variables taken from the filename pattern
#         xypzctr - Dimension to return (i.e. 'r' or 't')
#     Outputs:
#         index - The value of the variable
#     """


#     if xypzctr not in variables:
#         return 0
#     else:
#         return int(var_list[[ind for ind,v in zip(range(0,len(variables)),variables) if v==xypzctr][0]])

def get_nested(d):
    for a in d.keys():
        if isinstance(d[a],dict):
            for n in get_nested(d[a]):
                yield get_nested(n)
        else:
            yield d[a]

if __name__ == '__main__':
    a = {}
    a['a'] = {}
    a['a'][1] = 10
    a['a'][2] = 20
    a['b'] = {}
    a['b'][1] = 30
    a['b'][2] = 40

    for n in get_nested(a):
        print(n)