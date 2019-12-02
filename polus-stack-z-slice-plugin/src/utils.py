import re
from pathlib import Path

VARIABLES = 'pxyzctr'   # possible variables in input regular expression
STATICS = 'pxyctr'      # dimensions usually processed separately

def _parse_fpattern(fpattern):
    """ Parse the input filename pattern
    
    The filename pattern used here mimics that used by MIST, where variables and
    positions are encoded into the string. For example, file_c000.ome.tif that
    indicates channel using the _c, the filename pattern would be file_c{ccc}.ome.tif.
    The only possible variables that can be passed into the filename pattern are
    p, x, y, z, c, t, and r. In the case of p, x, and y, both x&y must be specified
    or p must be specified, but if all three are specified then an error is thrown.
    Inputs:
        fpattern - Input filename pattern
    Outputs:
        regex - Regex used to parse filenames
        variables - Variables found in the filename pattern
    """

    # Initialize the regular expression
    regex = fpattern

    # If no regex was supplied, return universal matching regex
    if fpattern==None or fpattern=='':
        return '.*', []
    
    # Parse variables
    expr = []
    variables = []
    for g in re.finditer(r"\{[pxyzctr]+\}",fpattern):
        expr.append(g.group(0))
        variables.append(expr[-1][1])
        
    # Verify variables are one of pxyzct
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

def _get_output_name(fpattern,file_ind,ind):
    """ Returns an output name for volumetric image
    This function returns a file output name for the image volume
    based on the names of the file names of the individual z-slices.
    All variables are kept the same as in the original filename,
    but the z values are transformed into a range surrounded by <>.
    For example, if the following files are processed:
    image_c000_z000.ome.tif
    image_c000_z001.ome.tif
    image_c000_z002.ome.tif
    then the output file will be:
    image_c000_z<000-002>.ome.tif
    Inputs:
        fpattern - A filename pattern indicating variables in filenames
        file_ind - A parsed dictionary of file names
        ind - A dictionary containing the indices for the file name (i.e. {'r':1,'t':1})
    Outputs:
        fname - an output file name
    """

    # If no regex was supplied, return default image name
    if fpattern==None or fpattern=='':
        return 'image.ome.tif'
    
    for key in ind.keys():
        assert key in VARIABLES, "Input dictionary key not a valid variable: {}".format(key)
    
    # Parse variables
    expr = []
    variables = []
    for g in re.finditer(r"\{[pxyzctr]+\}",fpattern):
        expr.append(g.group(0))
        variables.append(expr[-1][1])
        
    # Return an output file name
    fname = fpattern
    for e,v in zip(expr,variables):
        if v not in STATICS:
            minval = min([int(z) for z in file_ind.keys()])
            maxval = max([int(z) for z in file_ind.keys()])
            fname = fname.replace(e,'<' + str(minval).zfill(len(e)-2) +
                                    '-' + str(maxval).zfill(len(e)-2) + '>')
        elif v not in ind.keys():
            fname = fname.replace(e,str(0).zfill(len(e)-2))
        else:
            fname = fname.replace(e,str(ind[v]).zfill(len(e)-2))
        
    return fname

def _get_xypzctr(var_list,variables,xypzctr):
    """ Get the x, y, p, z, c, t, or r index
    
    When files are parsed, the variables are used in an index to provide
    a method to reference a specific file name by its dimensions. This
    function returns the variable index based on the input filename pattern.
    Inputs:
        var_list - List of values parsed from a filename using a filename pattern
        variables - List of permitted variables taken from the filename pattern
        zctr - Dimension to return (i.e. 'r' or 't')
    Outputs:
        index - The value of the dimension
    """
    if xypzctr not in variables:
        return 0
    else:
        return int(var_list[[ind for ind,v in zip(range(0,len(variables)),variables) if v==xypzctr][0]])

""" Parse files in an image collection according to a regular expression. """
def _parse_files_p(fpath,regex,variables):
    file_ind = {}
    files = [f.name for f in Path(fpath).iterdir() if f.is_file() and "".join(f.suffixes)=='.ome.tif']
    for f in files:
        groups = re.match(regex,f)
        if groups == None:
            continue
        
        # Get position variables
        p = _get_xypzctr(groups.groups(),variables,'p')
        z = _get_xypzctr(groups.groups(),variables,'z')
        t = _get_xypzctr(groups.groups(),variables,'t')
        c = _get_xypzctr(groups.groups(),variables,'c')
        r = _get_xypzctr(groups.groups(),variables,'r')
        
        if r not in file_ind.keys():
            file_ind[r] = {}
        if t not in file_ind[r].keys():
            file_ind[r][t] = {}
        if c not in file_ind[r][t].keys():
            file_ind[r][t][c] = {}
        if p not in file_ind[r][t][c].keys():
            file_ind[r][t][c][p] = {}
        if z not in file_ind[r][t][c][p].keys():
            file_ind[r][t][c][p][z] = []
            
        file_ind[r][t][c][p][z].append(f)
            
    return file_ind

def _parse_files_xy(fpath,regex,variables):
    file_ind = {}
    files = [f.name for f in Path(fpath).iterdir() if f.is_file() and "".join(f.suffixes)=='.ome.tif']
    for f in files:
        groups = re.match(regex,f)
        if groups == None:
            continue
        
        # Get position variables
        x = _get_xypzctr(groups.groups(),variables,'x')
        y = _get_xypzctr(groups.groups(),variables,'y')
        z = _get_xypzctr(groups.groups(),variables,'z')
        t = _get_xypzctr(groups.groups(),variables,'t')
        c = _get_xypzctr(groups.groups(),variables,'c')
        r = _get_xypzctr(groups.groups(),variables,'r')
        
        if r not in file_ind.keys():
            file_ind[r] = {}
        if t not in file_ind[r].keys():
            file_ind[r][t] = {}
        if c not in file_ind[r][t].keys():
            file_ind[r][t][c] = {}
        if x not in file_ind[r][t][c].keys():
            file_ind[r][t][c][x] = {}
        if y not in file_ind[r][t][c][x].keys():
            file_ind[r][t][c][x][y] = {}
        if z not in file_ind[r][t][c][x][y].keys():
            file_ind[r][t][c][x][y][z] = []
            
        file_ind[r][t][c][x][y][z].append(f)
            
    return file_ind
