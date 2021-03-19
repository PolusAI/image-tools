import argparse, logging, subprocess, time, multiprocessing, sys
import numpy as np
import os
import filepattern
import pandas as pd
import shutil
from pathlib import Path

def filter_planes(feature_dict, removeDirection, percentile):
    
    planes = list(feature_dict.keys())
    feat_value = [feature_dict[i] for i in planes]
    thresh = min(feat_value) + percentile * (max(feat_value) - min(feat_value))
    
    if removeDirection == 'Below':
        keep_planes = sorted([z for z in planes if feature_dict[z] >= thresh])
    else:
        keep_planes = sorted([z for z in planes if feature_dict[z] <= thresh])
    
    return set(keep_planes)

def make_uniform(planes_dict, uniques, delay):
    max_len = max([len(i) for i in planes_dict.values()])
    min_ind = min([min(planes_dict[k]) for k in planes_dict])
    max_ind = max([max(planes_dict[k]) for k in planes_dict])
    
    max_add_left = uniques.index(min_ind)
    max_add_right = len(uniques) - (uniques.index(max_ind)+1)
    
    for section_id, planes in planes_dict.items():
        len_to_add = max_len - len(planes)
        len_add_left = min(int(len_to_add)/2+delay, max_add_left)
        len_add_right = min(len_to_add - len_add_left+delay, max_add_right)
        left_ind = int(uniques.index(min(planes)) - len_add_left) 
        right_ind = int(uniques.index(max(planes)) + len_add_right)+1
        planes_dict[section_id] = uniques[left_ind:right_ind]
    return planes_dict

if __name__=="__main__":
    # Initialize the logger
    logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)

    ''' Argument parsing '''
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='Subset data using a given feature')
    
    # Input arguments
    parser.add_argument('--csvDir', dest='csvDir', type=str,
                        help='CSV collection containing features', required=True)
    parser.add_argument('--delay', dest='delay', type=str,
                        help='Number of images to capture outside the cutoff', required=True)
    parser.add_argument('--feature', dest='feature', type=str,
                        help='Feature to use to subset data', required=True)
    parser.add_argument('--filePattern', dest='filePattern', type=str,
                        help='Filename pattern used to separate data', required=True)
    parser.add_argument('--groupVar', dest='groupVar', type=str,
                        help='variables to group by in a section', required=True)
    parser.add_argument('--inpDir', dest='inpDir', type=str,
                        help='Input image collection to be processed by this plugin', required=True)
    parser.add_argument('--percentile', dest='percentile', type=str,
                        help='Percentile to remove', required=True)
    parser.add_argument('--removeDirection', dest='removeDirection', type=str,
                        help='remove direction above or below percentile', required=True)
    parser.add_argument('--sectionVar', dest='sectionVar', type=str,
                        help='variables to divide larger sections', required=True)
    parser.add_argument('--writeOutput', dest='writeOutput', type=str,
                        help='write output image collection or not', required=True)
    # Output arguments
    parser.add_argument('--outDir', dest='outDir', type=str,
                        help='Output collection', required=True)
    
    # Parse the arguments
    args = parser.parse_args()
    csvDir = args.csvDir
    if (Path.is_dir(Path(args.csvDir).joinpath('images'))):
        # switch to images folder if present
        fpath = str(Path(args.csvDir).joinpath('images').absolute())
    logger.info('csvDir = {}'.format(csvDir))
    delay = args.delay
    logger.info('delay = {}'.format(delay))
    feature = args.feature
    logger.info('feature = {}'.format(feature))
    filePattern = args.filePattern
    logger.info('filePattern = {}'.format(filePattern))
    groupVar = args.groupVar
    logger.info('groupVar = {}'.format(groupVar))
    inpDir = args.inpDir
    if (Path.is_dir(Path(args.inpDir).joinpath('images'))):
        # switch to images folder if present
        fpath = str(Path(args.inpDir).joinpath('images').absolute())
    logger.info('inpDir = {}'.format(inpDir))
    percentile = args.percentile
    logger.info('percentile = {}'.format(percentile))
    removeDirection = args.removeDirection
    logger.info('removeDirection = {}'.format(removeDirection))
    sectionVar = args.sectionVar
    logger.info('sectionVar = {}'.format(sectionVar))
    writeOutput = args.writeOutput == 'true'
    logger.info('writeOutput = {}'.format(writeOutput))
    outDir = args.outDir
    logger.info('outDir = {}'.format(outDir))
    
    # Surround with try/finally for proper error catching
    try:

        # Get all file names in csvDir image collection
        csvDir_files = [f.name for f in Path(csvDir).iterdir() if f.is_file() and "".join(f.suffixes)=='.csv']
        
        # Get all file names in inpDir image collection
        inpDir_files = [f.name for f in Path(inpDir).iterdir() if f.is_file() and "".join(f.suffixes)=='.ome.tif']

        # read and concat all csv files
        for ind, file in enumerate(csvDir_files):
            if ind == 0:
                feature_df = pd.read_csv(os.path.join(csvDir, file), header=None)
            else:
                feature_df = pd.concat([feature_df, pd.read_csv(os.path.join(csvDir, file), header=None)])
        
        # seperate filepattern variables into different categories
        _,var = filepattern.get_regex(filePattern)
        grouping_variables = groupVar.split(',')
        section_variables = sectionVar.split(',')
        sub_section_variables = [v for v in var if v not in grouping_variables+sectioning_variables]

        # initialize filepattern object
        fp = filepattern.FilePattern(inpDir, pattern=pattern)
        uniques = fp.uniques

        [maj_grouping_var, min_grouping_var] = grouping_variables if len(grouping_variables)>1 else grouping_variables+[None]
        keep_planes = {}


        # single iteration of this loop gives all images in one section
        for file in fp(group_by=sub_section_variables+grouping_variables):
            
            section_feat_dict = {}
            section_keep_planes = []
            section_id = tuple([file[0][i] for i in sectioning_variables]) if sectioning_variables[0] else 1
            
            # iterate over files in one section
            for f in file:
                if min_grouping_var == None:
                    f[min_grouping_var] == None
                
                # stote feature values for images 
                if f[min_grouping_var] not in section_feat_dict:
                    section_feat_dict[f[min_grouping_var]] = {}

                if f[maj_grouping_var] not in section_feat_dict[f[min_grouping_var]]:
                    section_feat_dict[f[min_grouping_var]][f[maj_grouping_var]] = []

                section_feat_dict[f[min_grouping_var]][f[maj_grouping_var]].append(feature_dict[f['file'].name])
            
            # average feature value by grouping variable
            for key1 in section_feat_dict:
                for key2 in section_feat_dict[key1]:
                    section_feat_dict[key1][key2] = sum(section_feat_dict[key1][key2])/len(section_feat_dict[key1][key2])
                
                # find planes to keep based on specified criteria
                section_keep_planes.append(filter_planes(section_feat_dict[key1],removeDirection, percentile))
            
            # keep same planes within a section, across the minor grouping variable
            section_keep_planes = list(section_keep_planes[0].union(*section_keep_planes))
            section_keep_planes = [i for i in range(min(section_keep_planes), max(section_keep_planes)+1) if i in uniques[maj_grouping_var]]
            keep_planes[section_id] = section_keep_planes
        
        # keep same number of planes across different sections
        keep_planes = make_uniform(keep_planes, uniques[maj_grouping_var], delay)
        
        # start writing summary.txt
        summary = open(os.path.join(outDir, 'summary.txt'), 'w')

        # rename subsetted data
        for file in fp(group_by=sub_section_variables+grouping_variables):
            section_id = tuple([file[0][i] for i in sectioning_variables]) if sectioning_variables[0] else 1
            section_keep_planes = keep_planes[section_id]
            summary.write('------------------------------------------------ \n')
            rename_map = {k:v for k,v in zip(keep_planes[section_id], uniques[maj_grouping_var])}
            for f in file:
                if f[maj_grouping_var] not in keep_planes[section_id]:
                    continue

                # olf and new file name
                old_file_name = f['file'].name
                file_name_dict = {k.upper():v for k,v in f.items() if k!='file'}
                file_name_dict[maj_grouping_var.upper()] = rename_map[f[maj_grouping_var]]
                new_file_name = fp.get_matching(**file_name_dict)[0]['file'].name

                # if write output collection
                if writeOutput:
                    shutil.move(os.path.join(inpDir, old_file_name),os.path.join(outDir, new_file_name))
                
                summary.write('{} -----> {} \n'.format(old_file_name, new_file_name))  
        summary.close()   

    finally:

        # Exit the program
        sys.exit()