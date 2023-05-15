import argparse
import logging 
import os
import csv
import numpy as np
from io import StringIO
import copy
from pathlib import Path
import logging
import vaex
import pandas as pd
import shutil
import functools as ft

POLUS_LOG = getattr(logging, os.environ.get('POLUS_LOG', 'INFO'))

FILE_EXT = os.environ.get('POLUS_EXT', '.csv')

if __name__=="__main__":
    # Initialize the logger
    logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)


    # Setup the argument parsing
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='Merge all csv files in a csv collection into a single csv file.')
    parser.add_argument('--inpDir', dest='inpDir', type=str,
                        help='Input image collection to be processed by this plugin', required=True)
    parser.add_argument('--stripExtension', dest='stripExtension', type=str,
                        help='Should csv be removed from the filename when indicating which file a row in a csv file came from?', required=True)
    parser.add_argument('--outDir', dest='outDir', type=str,
                        help='Output csv file', required=True)
    parser.add_argument('--dim', dest='dim', type=str,
                        help='rows or columns', required=True)
    parser.add_argument('--sameRows', dest='sameRows', type=str,
                        help='Only merge csvs if they contain the same number of rows', required=False)
    
        
    # Parse the arguments
    args = parser.parse_args()
    inpDir = args.inpDir
    logger.info('inpDir = {}'.format(inpDir))
    stripExtension = args.stripExtension == 'true'
    logger.info('stripExtension = {}'.format(stripExtension))
    outDir = args.outDir
    logger.info('outDir = {}'.format(outDir))
    dim = args.dim
    logger.info('dim = {}'.format(dim))
    same_rows = args.sameRows == 'true'
    logger.info('sameRows = {}'.format(stripExtension))
    
    # Get input files
    inpDir_files = [str(f.absolute()) for f in Path(inpDir).iterdir() if f.name.endswith('.csv')]
    inpDir_files.sort() # be a little fancy and merge alphabetically
    
    ''' If sameRows is set to true, nothing fancy to do. Just do the work and get out '''
    # Case One: If merging by columns and have same rows:
    if dim=='columns' and same_rows:
        logger.info("Merging data with identical number of rows...")
            
        # Determine the number of output files, and a list of files to be merged in each file
        out_files = {}
        for f in inpDir_files:
            with open(f,'r') as fr:
                count = sum(1 for row in fr)
            if count not in out_files.keys():
                out_files[count] = [f]
            else:
                out_files[count].append(f)
            
        count = 1
        for key in out_files.keys():
            
            outPath = str(Path(outDir).joinpath('merged_{}.csv'.format(count)).absolute())
            
            count += 1
            
            inp_files = [open(f) for f in out_files[key]]
            
            if FILE_EXT == ".feather":
                dfs = list()
                for l in range(key):
                    for f in inpDir_files:
                        df = pd.read_csv(f)
                        dfs.append(df)
                        df_final = ft.reduce(lambda left, right: pd.merge(left, right), dfs)
                        vaex_df = vaex.from_pandas(df_final)
                        vaex_df.export(outPath)
            else:
                with open(outPath,'w') as fw:
                    for l in range(key):
                        fw.write(','.join([f.readline().rstrip('\n') for f in inp_files]))
                        fw.write('\n')
                        
    else:
        # Get the column headers
        logger.info("Getting all unique headers...")
        headers = set([])
        identifiers = {}
        for in_file in inpDir_files:
            with open(in_file,'r') as fr:
                
                # Get header information
                line = fr.readline()
                
                if dim=='columns' and 'file' not in line[0:-1].split(','):
                    ValueError('No file columns found in csv: {}'.format(in_file))
                
                h = line.rstrip('\n').split(',')
                headers.update(h)
                
                # Check to see if column identifiers for Plots API exist in the 2nd row
                line = fr.readline()
                ident = line.rstrip('\n').split(',')
                no_identifier = sum(1 for f in ident if f not in 'FC')
                               
        if 'file' in headers:
            headers.remove('file')
        headers = list(headers)
        headers.sort()
        headers.insert(0,'file')
        if identifiers:
            for h in headers:
                if h not in identifiers.keys():
                    identifiers[h] = 'F'
        logger.info("Unique headers: {}".format(headers))

        # Generate the line template
        line_template = ','.join([('{' + h + '}') for h in headers]) + '\n'
        line_dict = {key:'NaN' for key in headers}
        
        # Generate the path to the output file
        outPath = str(Path(outDir).joinpath('merged.feather').absolute()) if FILE_EXT == 'feather' else str(Path(outDir).joinpath('merged.csv').absolute())
        
        # Case Two: Merger along rows only
        if dim=='rows':
            logger.info("Merging the data along rows...")
            with open(outPath,'w') as out_file:
                out_file.write(','.join(headers) + '\n')
                if identifiers:
                    out_file.write(line_template.format(**identifiers))
                for f in inpDir_files:
                    file_dict = copy.deepcopy(line_dict)
                    if stripExtension:
                        file_dict['file'] = str(Path(f).name).replace('.csv','')
                    else:
                        file_dict['file'] = str(Path(f).name)
                    logger.info("Merging file: {}".format(file_dict['file']))

                    with open(f,'r') as in_file:
                        file_map = in_file.readline().rstrip('\n')
                        file_map = file_map.split(',')
                        numel = len(file_map)
                        file_map = {ind:key for ind,key in enumerate(file_map)}
                        
                        # Check to see if column identifiers are present in the file, skip 2nd row if they are present
                        if identifiers:
                            line = in_file.readline()
                            ident = line.rstrip('\n').split(',')
                            no_identifier = sum(1 for f in ident if f not in 'FC')
                            in_file.seek(0)
                            in_file.readline()
                            if not no_identifier:
                                in_file.readline()

                        for line in in_file:
                            for el,val in enumerate(line.rstrip('\n').split(',')):
                                file_dict[file_map[el]] = val
                            out_file.write(line_template.format(**file_dict))
                            
            
            # Write Merged file
            if FILE_EXT == '.feather':
                logger.info("Merging the data along rows for feather file")
                temp_df = pd.read_csv(outPath)
                df = vaex.from_pandas(temp_df)
                os.chdir(outDir)
                df.export(outPath)
            
        # Case Three: Merger along columns only
        elif dim=='columns':
            logger.info("Merging the data along columns...")
            outPath = str(Path(outDir).joinpath('merged.csv').absolute())
            
            # Load the first csv and generate a dictionary to hold all values
            out_dict = {}
            with open(inpDir_files[0],'r') as in_file:
                file_map = in_file.readline().rstrip('\n')
                file_map = file_map.split(',')
                numel = len(file_map)
                file_map = {ind:key for ind,key in zip(range(numel),file_map)}
                
                for line in in_file:
                    file_dict = copy.deepcopy(line_dict)
                    for el,val in enumerate(line.rstrip('\n').split(',')):
                        file_dict[file_map[el]] = val
                    if file_dict['file'] in out_dict.keys():
                        UserWarning('Skipping row for file since it is already in the output file dictionary: {}'.format(file_dict['file']))
                    else:
                        out_dict[file_dict['file']] = file_dict
                        
            # Loop through the remaining files and update the output dictionary
            for f in inpDir_files[1:]:
                
                with open(f,'r') as in_file:
                    file_dict = copy.deepcopy(line_dict)
                    file_map = in_file.readline().rstrip('\n')
                    file_map = file_map.split(',')
                    file_ind = [i for i,v in enumerate(file_map) if v == 'file'][0]
                    numel = len(file_map)
                    file_map = {ind:key for ind,key in zip(range(numel),file_map)}
                    
                    for line in in_file:
                        file_dict = copy.deepcopy(line_dict)
                        line_vals = line.rstrip('\n').split(',')
                        for el,val in enumerate(line_vals):
                            if line_vals[file_ind] not in out_dict.keys():
                                out_dict[line_vals[file_ind]] = copy.deepcopy(line_dict)
                                out_dict[line_vals[file_ind]]['file'] = line_vals[file_ind]
                            if el == file_ind:
                                continue
                            if out_dict[line_vals[file_ind]][file_map[el]] != 'NaN':
                                Warning('Skipping duplicate value ({}) found in {}'.format(val,in_file))
                            else:
                                out_dict[line_vals[file_ind]][file_map[el]] = val
                                
            # Write the output file using ENV Variable
            with open(outPath,'w') as out_file:
            # Write headers
                out_file.write(','.join(headers) + '\n')
            
                for val in out_dict.values():
                    out_file.write(line_template.format(**val))
                    
            # Write Merged Feather by reading lines into dataframe
            if FILE_EXT == '.feather':
                df = pd.DataFrame.from_dict(out_dict, orient='index')
                vaex_df = vaex.from_pandas(df)
                vaex_df.export(outPath)