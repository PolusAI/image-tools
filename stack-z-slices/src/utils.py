import re
from pathlib import Path

VARIABLES = 'pxyzct'   # possible variables in input regular expression
STATICS = 'zt'         # dimensions usually processed separately

# Initialize the logger

""" Parse a regular expression given by the plugin """
def _parse_regex(regex):
    # If no regex was supplied, return universal matching regex
    if regex==None or regex=='':
        return '.*'
    
    # Parse variables
    expr = []
    variables = []
    for g in re.finditer(r"\{[pxyzct]+\}",regex):
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
        
    # Return a regular expression pattern
    for e in expr:
        regex = regex.replace(e,"([0-9]{"+str(len(e)-2)+"})")
        
    return regex, variables

def _get_output_name(regex,ind):
    # If no regex was supplied, return default image name
    if regex==None or regex=='':
        return 'image.ome.tif'
    
    for key in ind.keys():
        assert key in VARIABLES, "Input dictionary key not a valid variable: {}".format(key)
    
    # Parse variables
    expr = []
    variables = []
    for g in re.finditer(r"\{[pxyzct]+\}",regex):
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
        
    # Return a regular expression pattern
    for e,v in zip(expr,variables):
        if v in 'xyp' or v not in ind.keys():
            regex = regex.replace(e,str(0).zfill(len(e)-2))
        else:
            regex = regex.replace(e,str(ind[v]).zfill(len(e)-2))
        
    return regex

""" Get the z, c, or t variable if it exists. Return 0 otherwise. """
def _get_xyzct(var_list,variables,xyzct):
    if xyzct not in variables:
        return 0
    else:
        return int(var_list[[ind for ind,v in zip(range(0,len(variables)),variables) if v==xyzct][0]])

""" Parse files in an image collection according to a regular expression. """
def _parse_files_p(fpath,regex,variables):
    file_ind = {}
    files = [f.name for f in Path(fpath).iterdir() if f.is_file() and "".join(f.suffixes)=='.ome.tif']
    for f in files:
        groups = re.match(regex,f)
        if groups == None:
            continue
        
        # Get position variables
        p = _get_xyzct(groups.groups(),variables,'p')
        z = _get_xyzct(groups.groups(),variables,'z')
        t = _get_xyzct(groups.groups(),variables,'t')
        c = _get_xyzct(groups.groups(),variables,'c')
        
        if t not in file_ind.keys():
            file_ind[t] = {}
        if c not in file_ind[t].keys():
            file_ind[t][c] = {}
        if p not in file_ind[t][c].keys():
            file_ind[t][c][p] = {}
        if z not in file_ind[t][c][p].keys():
            file_ind[t][c][p][z] = []
            
        file_ind[t][c][p][z].append(f)
            
    return file_ind

def _parse_files_xy(fpath,regex,variables):
    file_ind = {}
    files = [f.name for f in Path(fpath).iterdir() if f.is_file() and "".join(f.suffixes)=='.ome.tif']
    for f in files:
        groups = re.match(regex,f)
        if groups == None:
            continue
        
        # Get position variables
        x = _get_xyzct(groups.groups(),variables,'x')
        y = _get_xyzct(groups.groups(),variables,'y')
        z = _get_xyzct(groups.groups(),variables,'z')
        t = _get_xyzct(groups.groups(),variables,'t')
        c = _get_xyzct(groups.groups(),variables,'c')
        
        if t not in file_ind.keys():
            file_ind[t] = {}
        if c not in file_ind[t].keys():
            file_ind[t][c] = {}
        if x not in file_ind[t][c].keys():
            file_ind[t][c][x] = {}
        if y not in file_ind[t][c][x].keys():
            file_ind[t][c][x][y] = {}
        if z not in file_ind[t][c][x][y].keys():
            file_ind[t][c][x][y][z] = []
            
        file_ind[t][c][x][y][z].append(f)
            
    return file_ind
