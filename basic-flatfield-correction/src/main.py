import argparse
import os
import time
import csv
from pathlib import Path
import logging
import re
from bfio.bfio import BioReader,BioWriter

import numpy as np

VARIABLES = 'pxyzct'   # possible variables in input regular expression
STATICS = 'zt'         # dimensions usually processed separately

""" Parse a regular expression given by the plugin """
def _parse_regex(regex):
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

""" Get the z, c, or t variable if it exists. Return 0 otherwise. """
def _get_zct(var_list,variables,zct):
    if zct not in variables:
        return 0
    else:
        return int(var_list[[ind for ind,v in zip(range(0,len(variables)),variables) if v==zct][0]])

""" Parse files in an image collection according to a regular expression. """
def _parse_files(fpath,regex,variables):
    file_ind = {}
    files = [f.name for f in Path(fpath).iterdir() if f.is_file() and "".join(f.suffixes)=='.ome.tif']
    for f in files:
        groups = re.match(regex,f)
        if groups == None:
            continue
        z = _get_zct(groups.groups(),variables,'z')
        t = _get_zct(groups.groups(),variables,'t')
        c = _get_zct(groups.groups(),variables,'c')
        if z not in file_ind.keys():
            file_ind[z] = {}
        if t not in file_ind[z].keys():
            file_ind[z][t] = {}
        if c not in file_ind[z][t].keys():
            file_ind[z][t][c] = {'file': []}
            file_ind[z][t][c].update({key:[] for key in variables if key not in STATICS})
        file_ind[z][t][c]['file'].append(f)
        for key,group in zip(variables,groups.groups()):
            if key in STATICS:
                continue
            elif key in VARIABLES:
                group = int(group)
            file_ind[z][t][c][key].append(group)
            
    return file_ind

""" Load files and create an image stack """
def _get_image_stack(fpath,regex):
    
    # Get images matching the regex
    pass

def main():
    """ Initialize the logger """
    logging.basicConfig(format='%(asctime)s - %(name)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    """ Initialize argument parser """
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(prog='s3import', description='Pull data from an S3 bucket.')

    """ Define the arguments """
    parser.add_argument('--s3Bucket',               # Name of the bucket
                        dest='s3_bucket',
                        type=str,
                        help='S3 bucket',
                        required=True)
    parser.add_argument('--s3Key',                  # Path to the data within the bucket
                        dest='s3_key',
                        type=str,
                        help='S3 bucket',
                        required=True)
    parser.add_argument('--awsAccessKey',           # AWS access key
                        dest='aws_access_key',
                        type=str,
                        help='The AWS access key id used to access the S3 bucket',
                        required=True)
    parser.add_argument('--awsAccessSecret',        # AWS secret key
                        dest='aws_access_secret',
                        type=str,
                        help='The AWS secret access key used to access the S3 bucket',
                        required=True)
    parser.add_argument('--outDir',                 # Output directory
                        dest='output_dir',
                        type=str,
                        help='The output directory for the data pulled from S3',
                        required=True)
    parser.add_argument('--getMeta',                # Get metadata instead of bioformats data
                        dest='get_metadata',
                        type=str,
                        help='If true, grabs metadata rather than images',
                        required=False)
    
    """ Get the input arguments """
    args = parser.parse_args()

    s3_bucket = args.s3_bucket
    s3_key = args.s3_key
    aws_access_key = args.aws_access_key
    aws_access_secret = args.aws_access_secret
    output_dir = args.output_dir
    get_metadata = args.get_metadata=='true'

    # If the key ends with a file separator, no files will be downloaded.
    # Remove trailing file separator if present.
    if s3_key[-1] == os.path.sep:
        s3_key = s3_key[0:-1]

    logger.info('s3_bucket = {}'.format(s3_bucket))
    logger.info('s3_key = {}'.format(s3_key))
    logger.info('aws_access_key = {}'.format(aws_access_key))
    logger.info('aws_access_secret = {}'.format(aws_access_secret))
    logger.info('output_dir = {}'.format(output_dir))
    logger.info('get_metadata = {}'.format(get_metadata))

    """ Initialize the S3 client """
    logger.info('Initializing boto3 client...')
    try:
        client = boto3.session.Session(aws_access_key_id=aws_access_key,
                                       aws_secret_access_key=aws_access_secret)    
        s3 = client.resource('s3')
        bucket = s3.Bucket(s3_bucket)
    except Exception:
        logger.exception("Failed to create an S3 session.")
    logger.info('Client initialized! Starting file transfer...')
    
    """ Download files """
    download_start = time.time()
    all_files = bucket.objects.all()
    for f in all_files:
        p, fname = os.path.split(f.key)
        
        # If the name is blank, it's a directory so skip
        if fname=="":
            continue
        # If the directory does not match the input, skip it
        if p!=s3_key:
            continue
            
        if p==s3_key and fname != "":
            try:
                logger.info('{} >>> {}'.format(fname,output_dir + os.path.sep + fname))
                bucket.Object(f.key).download_file(output_dir + os.path.sep + fname)
            except Exception as e:
                logger.exception("Failed to download file: {}".format(fname))
    
    logger.info('All files downloaded in {} seconds!'.format(time.time() - download_start))
        
if __name__ == "__main__":
    fpath = "/mnt/3cc68e90-9b42-43df-b244-3492e361b382/WIPP/WIPP-plugins/collections/5d9bea5dd46a8d000104f273/images"
    inp_regex = "S1_R{t}_C1-C11_A1_y{yyy}_x{xxx}_c{ccc}.ome.tif"
    
    regex,variables = _parse_regex(inp_regex)
    print("(old regex,new regex) -> ({},{})".format(inp_regex,regex))
    files = _parse_files(fpath,regex,variables)
    print(files[0][1][0]['file'])
    #main()