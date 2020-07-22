import argparse, logging, subprocess, time, multiprocessing
import numpy as np
from pathlib import Path
from parser import parse_collection
import os
import psutil
import shutil


if __name__=="__main__":
    # Initialize the logger
    logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)

    # Setup the argument parsing
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='This plugin registers an image collection')
    parser.add_argument('--filePattern', dest='filePattern', type=str, help='Filename pattern used to separate data', required=True)
    parser.add_argument('--inpDir', dest='inpDir', type=str,help='Input image collection to be processed by this plugin', required=True)
    parser.add_argument('--registrationVariable', dest='registrationVariable', type=str, help='variable to help identify which images need to be registered to each other', required=True)
    parser.add_argument('--template', dest='template', type=str, help='Template image to be used for image registration', required=True)
    parser.add_argument('--TransformationVariable', dest='TransformationVariable', type=str,help='variable to help identify which images have similar transformation', required=True)
    parser.add_argument('--outDir', dest='outDir', type=str, help='Output collection', required=True)
    
    
    # Parse the arguments     
    args = parser.parse_args()
    filePattern = args.filePattern
    logger.info('filePattern = {}'.format(filePattern))
    
    inpDir = args.inpDir
    # check if images folder is present in the input directory
    if (Path.is_dir(Path(inpDir).joinpath('images'))):
        inpDir= str(Path(inpDir).joinpath('images'))
        
    logger.info('inpDir = {}'.format(inpDir))
    registrationVariable = args.registrationVariable
    logger.info('registrationVariable = {}'.format(registrationVariable))
    template = args.template
    logger.info('template = {}'.format(template))
    TransformationVariable = args.TransformationVariable
    logger.info('TransformationVariable = {}'.format(TransformationVariable))
    outDir = args.outDir
    logger.info('outDir = {}'.format(outDir))     
    

        

    #memory usage    
    mem = psutil.virtual_memory()
    logger.debug('System memory stats : {}'.format(mem))  
    
     # get template image path
    template_image_path=os.path.join(inpDir,template)  
    
    # filename len
    filename_len= len(template)
     
    # parse the input collection
    logger.info('Parsing the input collection and getting registration_dictionary')
    registration_dictionary=parse_collection(inpDir,filePattern,registrationVariable, TransformationVariable, template_image_path)
    
    logger.info('Iterating over registration_dictionary....')
    for registration_set,similar_transformation_set in registration_dictionary.items():
        
        # registration_dictionary consists of set of already registered images as well
        if registration_set[0]==registration_set[1]:            
            similar_transformation_set=similar_transformation_set.tolist()
            similar_transformation_set.append(registration_set[0])
            for image_path in similar_transformation_set:
                image_name=image_path[-1*filename_len:]
                logger.info('Copying image {} to output directory'.format(image_name))
                shutil.copy2(image_path,os.path.join(outDir,image_name))            
            continue
        
        # concatenate lists into a string to pass as an argument to argparse
        registration_string=' '.join(registration_set)
        similar_transformation_string=' '.join(similar_transformation_set)        

        # open subprocess image_registration.py
        registration = subprocess.Popen("python3 image_registration.py --registrationString '{}' --similarTransformationString '{}' --outDir '{}' --template '{}'".format(registration_string,similar_transformation_string,outDir,template ), shell=True )
        registration.wait()
        