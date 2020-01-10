import argparse
import time
from pathlib import Path
import os
import logging
import re

STITCH_VARS = ['file','correlation','posX','posY','gridX','gridY'] # image stitching values
VARIABLES = 'pxyzctr'                                              # possible variables in input filename pattern

# The variables z and t must match across different images in order for an existing stitching
# vector to be applied to them. If neither z nor t is supplied in the stitching vector regex,
# then they will be ignored. 
STATICS = 'zt'

""" Parse a regular expression given by the plugin """
def _parse_regex(regex):
    # Parse variables
    expr = []
    variables = []
    for g in re.finditer(r"\{[pxyzctr]+\}",regex):
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
        
    # Return a regular expression pattern
    for e in expr:
        regex = regex.replace(e,"([0-9]{"+str(len(e)-2)+"})")
        
    return regex, variables

""" Load and parse image stitching vectors """
def _parse_stitch(stitchPath,regex,variables):
    # Get the stitch files
    txt_files = [f.name for f in Path(stitchPath).iterdir() if f.is_file() and f.suffix=='.txt']
    global_regex = ".*-global-positions-[0-9]+.txt"
    stitch_vectors = [re.match(global_regex,f).group(0) for f in txt_files if re.match(global_regex,f) is not None]
    
    line_regex = r"file: (.*); corr: (.*); position: \((.*), (.*)\); grid: \((.*), (.*)\);"
    variables.extend(STITCH_VARS)
    stitch_ind = {}

    for vector in stitch_vectors:
        fpath = os.path.join(stitchPath,vector)
        with open(fpath,'r') as fr:
            # Get the first line to determine z and t
            line = fr.readline()
            stitch_groups = re.match(line_regex,line)
            stitch_groups = list(stitch_groups.groups())
            groups = re.match(regex,stitch_groups[0])
            groups = list(groups.groups())
            z = _get_zctr(groups,variables,'z')
            t = _get_zctr(groups,variables,'t')
            r = _get_zctr(groups,variables,'r')
            if r not in stitch_ind.keys():
                stitch_ind[r] = {}
            if z not in stitch_ind[r].keys():
                stitch_ind[r][z] = {}
            if t not in stitch_ind[r][z].keys():
                stitch_ind[r][z][t] = {}
                stitch_ind[r][z][t].update({key:[] for key in variables if key not in STATICS})
            fr.seek(0)
            
            # Parse all lines of the stitch vector
            for line in fr:
                stitch_groups = re.match(line_regex,line)
                stitch_groups = list(stitch_groups.groups())
                groups = re.match(regex,stitch_groups[0])
                groups = list(groups.groups())
                groups.extend(stitch_groups)
                for key,group in zip(variables,groups):
                    if key in STATICS:
                        continue
                    if key in VARIABLES:
                        group = int(group)
                    stitch_ind[r][z][t][key].append(group)
                    
    return stitch_ind

""" Get the z, c, t, or r variable if it exists. Return 0 otherwise. """
def _get_zctr(var_list,variables,zctr):
    if zctr not in variables:
        return 0
    else:
        return int(var_list[[ind for ind,v in zip(range(0,len(variables)),variables) if v==zctr][0]])

""" Parse files in an image collection according to a regular expression. """
def _parse_files(fpath,regex,variables):
    file_ind = {}
    files = [f.name for f in Path(fpath).iterdir() if f.is_file() and "".join(f.suffixes)=='.ome.tif']
    for f in files:
        groups = re.match(regex,f)
        if groups == None:
            continue
        z = _get_zctr(groups.groups(),variables,'z')
        t = _get_zctr(groups.groups(),variables,'t')
        c = _get_zctr(groups.groups(),variables,'c')
        r = _get_zctr(groups.groups(),variables,'r')
        if r not in file_ind.keys():
            file_ind[r] = {}
        if z not in file_ind[r].keys():
            file_ind[r][z] = {}
        if t not in file_ind[r][z].keys():
            file_ind[r][z][t] = {}
        if c not in file_ind[r][z][t].keys():
            file_ind[r][z][t][c] = {'file': []}
            file_ind[r][z][t][c].update({key:[] for key in variables if key not in STATICS})
        file_ind[r][z][t][c]['file'].append(f)
        for key,group in zip(variables,groups.groups()):
            if key in STATICS:
                continue
            elif key in VARIABLES:
                group = int(group)
            file_ind[r][z][t][c][key].append(group)
            
    return file_ind

def _validate_vars(stitch_vars,file_vars):
    for v in stitch_vars:
        assert v in file_vars, "Variable {} is in stitch regular expression but not file regular expression.".format(v)
    for v in file_vars:
        assert v in stitch_vars, "Variable {} is in file regular expression but not stitch regular expression.".format(v)
        
def _get_ind_p(stitch_index,file_index,index):
    for i in range(0,len(file_index['p'])):
        if stitch_index['p'][index] == file_index['p'][i]:
            return i
    return None

def _get_ind_xy(stitch_index,file_index,index):
    for i in range(0,len(file_index['x'])):
        if stitch_index['x'][index] == file_index['x'][i] and stitch_index['y'][index] == file_index['y'][i]:
            return i
    return None

def _generate_stitch(fpath,stitch_vector,file_vector):
    if 'p' in file_vector.keys():
        get_ind = _get_ind_p
    else:
        get_ind = _get_ind_xy
    
    with open(fpath,'w') as fw:
        for i in range(0,len(stitch_vector['posX'])):
            ind = get_ind(stitch_vector,file_vector,i)
            # If there isn't a matching stitching index in the files, move to the next stitch index
            if ind == None:  
                continue
            fw.write("file: {}; corr: {}; position: ({}, {}); grid: ({}, {});\n".format(file_vector['file'][ind],
                                                                                        stitch_vector['correlation'][i],
                                                                                        stitch_vector['posX'][i],
                                                                                        stitch_vector['posY'][i],
                                                                                        stitch_vector['gridX'][i],
                                                                                        stitch_vector['gridY'][i]))
        
def main():
    # Initialize the logger
    logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)
    
    # Setup the Argument parsing
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='Extract individual fields of view from a czi file.')

    parser.add_argument('--stitchDir', dest='stitch_dir', type=str,
                        help='Path to folder with CZI files', required=True)
    parser.add_argument('--collectionDir', dest='collection_dir', type=str,
                        help='Path to folder with CZI files', required=True)
    parser.add_argument('--stitchRegex', dest='stitch_regex', type=str,
                        help='Path to folder with CZI files', required=True)
    parser.add_argument('--collectionRegex', dest='collection_regex', type=str,
                        help='Path to folder with CZI files', required=True)
    parser.add_argument('--outDir', dest='output_dir', type=str,
                        help='The directory in which to save stitching vectors.', required=True)

    # Get the arguments
    args = parser.parse_args()
    stitch_dir = args.stitch_dir
    collection_dir = args.collection_dir
    stitch_regex = args.stitch_regex
    collection_regex = args.collection_regex
    output_dir = args.output_dir
    logger.info('stitch_dir = {}'.format(stitch_dir))
    logger.info('collection_dir = {}'.format(collection_dir))
    logger.info('stitch_regex = {}'.format(stitch_regex))
    logger.info('collection_regex = {}'.format(collection_regex))
    logger.info('output_dir = {}'.format(output_dir))
    
    # Parse the regular expression from the file name pattern
    logger.info("Parsing regular expressions...")
    new_stitch_regex, stitch_variables = _parse_regex(stitch_regex)
    new_collection_regex, collection_variables = _parse_regex(collection_regex)
    logger.info("Parsed stitch regex: {}".format(new_stitch_regex))
    logger.info("Parsed collection regex: {}".format(new_collection_regex))
    
    # Validate variables - make sure the only variables are x,y,z,c,t,r
    logger.info("Validating variables...")
    _validate_vars(stitch_variables,collection_variables)
    logger.info("Passed variable checks.")
    
    # Parse all files in the stitching vector directory to get position information
    logger.info("Parsing stitching vectors...")
    stitch_vector = _parse_stitch(stitch_dir,new_stitch_regex,stitch_variables)
    logger.info("Found the following [r][z][t] combinations:")
    rs = [key for key in stitch_vector.keys()]
    rs.sort()
    for r in rs:
        zs = [key for key in stitch_vector[r].keys()]
        zs.sort()
        for z in zs:
            ts = [key for key in stitch_vector[r][z].keys()]
            ts.sort()
            for t in ts:
                logger.info("[r][z][t]: [{}][{}][{}]".format(r,z,t))
    
    # Parse all files in the image collection directory to get position, channel, etc
    logger.info("Parsing files in collection...")
    collection_index = _parse_files(collection_dir,new_collection_regex,collection_variables)
    logger.info("Found the following [r][z][t][c] combinations:")
    rs = [key for key in stitch_vector.keys()]
    rs.sort()
    for r in rs:
        zs = [key for key in collection_index[r].keys()]
        zs.sort()
        for z in zs:
            ts = [key for key in collection_index[r][z].keys()]
            ts.sort()
            for t in ts:
                cs = [key for key in collection_index[r][z][t].keys()]
                cs.sort()
                for c in cs:
                    logger.info("[r][z][t][c]: [{}][{}][{}][{}]".format(r,z,t,c))
    
    # Use the r,z,t information from the existing stitching vectors to generate new stitching vectors
    # using image names from the image collection.
    # The if-statements below are used to catch instances where a value for a variable is present in
    # the stitching vectors but not in the image collections. For example, if there is a vector where
    # r=1 in the stitching vector but there is no image set with r=1, then move to the next vector.
    logger.info("Generating stitching vectors...")
    out_index = 1
    rs = [key for key in stitch_vector.keys()]
    rs.sort()
    for r in rs:
        if r not in collection_index.keys():
            continue
        zs = [z for z in stitch_vector[r].keys()]
        zs.sort()
        for z in zs:
            if z not in collection_index[r].keys():
                continue
            ts = [t for t in stitch_vector[r][z].keys()]
            ts.sort()
            for t in ts:
                if t not in collection_index[r][z].keys():
                    continue
                cs = [c for c in collection_index[r][z][t].keys()]
                cs.sort()
                for c in cs:
                    fname = "img-global-positions-{}.txt".format(out_index)
                    logger.info("Generating vector: {}".format(fname))
                    fout = os.path.join(output_dir,fname)
                    out_index += 1
                    _generate_stitch(fout,stitch_vector[r][z][t],collection_index[r][z][t][c])
                
    logger.info("Plugin completed all operations!")

if __name__ == "__main__":
    main()