""" utils.py - general file io functions """

import re
from pathlib import Path

STATICS = 'ctr'         # dimensions usually processed separately
VARIABLES = 'pxyzctr'   # possible variables in input regular expression

def _get_zctr(var_list,variables,zctr):
    """ Get the z, c, t, or r index
    
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
    if zctr not in variables:
        return 0
    else:
        return int(var_list[[ind for ind,v in zip(range(0,len(variables)),variables) if v==zctr][0]])

""" Parse a regular expression given by the plugin """
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

""" Parse files in an image collection according to a regular expression. """
def _parse_files(fpath,regex,variables):
    """ Parse filenames into a dictionary according to input variables
    
    This function takes a list of input filenames, extracts the variables
    from each filename, and returns a dictionary where keys are the variable
    values for each filename. The dimension order is r, t, c. The value for
    each r, t, c position is a list of filenames that may contain different
    x, y, p, or z positions. This is because the BaSiC flatfield algorithm
    tries to estimate uneven illumination per channel independent of physical
    position. Replicate and timepoint are separate from position since it
    may be desired to recalculate the flatfield for each imaging session due
    to environmental variables (e.g. room/illumination source temperature or
    age).

    Inputs:
        fpath - Input path containing images
        regex - A regular expression with groups corresponding to each variable
        variables - A list of variables in the regular expression
    Outputs:
        file_ind - A dictionary containing dimensions as keys and filenames as values
    """
    file_ind = {}
    files = [f.name for f in Path(fpath).iterdir() if f.is_file() and "".join(f.suffixes)=='.ome.tif']
    for f in files:
        groups = re.match(regex,f)
        if groups == None:
            continue
        r = _get_zctr(groups.groups(),variables,'r')
        t = _get_zctr(groups.groups(),variables,'t')
        c = _get_zctr(groups.groups(),variables,'c')
        if r not in file_ind.keys():
            file_ind[r] = {}
        if t not in file_ind[r].keys():
            file_ind[r][t] = {}
        if c not in file_ind[r][t].keys():
            file_ind[r][t][c] = {'file': []}
            file_ind[r][t][c].update({key:[] for key in variables if key not in STATICS})
        file_ind[r][t][c]['file'].append(f)
        for key,group in zip(variables,groups.groups()):
            if key in STATICS:
                continue
            elif key in VARIABLES:
                group = int(group)
            file_ind[r][t][c][key].append(group)
            
    return file_ind

# TODO: Finish this function, currently does not do the <> notation
def _get_output_name(fpattern,file_ind,ind):
    """ Returns an output name for the flatfield or darkfield image

    This function returns a file output name for the flatfield/darkfield
    images based on the files used to calculate the flatfield/darkfield
    values. The variables r, t, and c are kept the same as in the
    original filename, but the variables x, y, z, and p are transformed
    into a range surrounded by <>. For example, if the following files
    are processed:
    image_x000_y000_c000.ome.tif
    image_x001_y000_c000.ome.tif
    image_x000_y001_c000.ome.tif
    image_x001_y001_c000.ome.tif

    then the output file will be:
    image_x<000-001>_y<000-001>_c000.ome.tif

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
        if v in 'xyzp':
            minval = min(file_ind[v])
            maxval = max(file_ind[v])
            fname = fname.replace(e,'<' + str(minval).zfill(len(e)-2) +
                                    '-' + str(maxval).zfill(len(e)-2) + '>')
        elif v not in ind.keys():
            fname = fname.replace(e,str(0).zfill(len(e)-2))
        else:
            fname = fname.replace(e,str(ind[v]).zfill(len(e)-2))
        
    return fname