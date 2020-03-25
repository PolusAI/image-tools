from bfio.bfio import BioReader, BioWriter
import bioformats
import javabridge as jutil
import argparse, logging, subprocess, time, multiprocessing
import numpy as np
from pathlib import Path
from collection_parser import collection_parser
from image_registration import apply_registration, register_images
import os


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
    logger.info('inpDir = {}'.format(inpDir))
    registrationVariable = args.registrationVariable
    logger.info('registrationVariable = {}'.format(registrationVariable))
    template = args.template
    logger.info('template = {}'.format(template))
    TransformationVariable = args.TransformationVariable
    logger.info('TransformationVariable = {}'.format(TransformationVariable))
    outDir = args.outDir
    logger.info('outDir = {}'.format(outDir))
    
    
    # Start the javabridge with proper java logging
    logger.info('Initializing the javabridge...')
    log_config = Path(__file__).parent.joinpath("log4j.properties")
    jutil.start_vm(args=["-Dlog4j.configuration=file:{}".format(str(log_config.absolute()))],class_path=bioformats.JARS)
    
    # get template image path
    template_image_path=os.path.join(inpDir,template)
    
    # parse the input collection
    logger.info('Parsing the input collection and getting registration_dictionary')
    registration_dictionary=collection_parser(inpDir,filePattern,registrationVariable, TransformationVariable, template_image_path)
    
    filename_len=len(template)
    # Loop through files in inpDir image collection and process
    
    logger.info('Iterating over all the registration sets....')
    for registration_set,similar_transformation_set in registration_dictionary.items():
        
        # registration_dictionary consists of set of already registered images as well
        if registration_set[0]==registration_set[1]:
            continue
        
        logger.info('calculating transformation to register image {} to {}'.format(registration_set[1],registration_set[0]))        
        #read template image for a particular set
        bf = BioReader(registration_set[0])
        template_img = bf.read_image()
        template_img=template_img[:,:,0,0,0]
        
        # read moving image for a particular set
        bf = BioReader(registration_set[1])
        moving_img = bf.read_image()    
        moving_img=moving_img[:,:,0,0,0]
        
        # seperate the filename of the moving image from the complete path
        moving_image_name=registration_set[1][-1*filename_len:]        
        
        # register the images and store the set of transformation used to carry out the registration
        transformed_moving_image, Rough_Homography_Upscaled, fine_homography_set=register_images(template_img, moving_img)
        
        logger.info('transformation calculated and writing output of {}'.format(moving_image_name))
        # write the output to the desired directory using bfio
        transformed_moving_image_5channel=np.zeros((transformed_moving_image.shape[0],transformed_moving_image.shape[1],1,1,1),dtype='uint16')
        transformed_moving_image_5channel[:,:,0,0,0]=transformed_moving_image               
        bw = BioWriter(os.joinpath(outDir,moving_image_name ), metadata=bf.read_metadata() )
        bw.write_image(transformed_moving_image_5channel)
        bw.close_image() 
        
        # delete to free memory        
        del transformed_moving_image 
        del transformed_moving_image_5channel        
        
        # iterate across all images which have the similar transformation as the moving image above  
        logger.info('iterate over all images that have similar transformation')
        for image_name in similar_transformation_set:
            
            # seperate image name from the path to it
            moving_image_name=image_name[-1*filename_len:]            
            
            logger.info('applying transformation to image {}'.format(moving_image_name))
            # read moving image using bfio
            bf = BioReader(image_name)
            moving_img = bf.read_image()    
            moving_img=moving_img[:,:,0,0,0] 
            
            # use the precalculated transformation to transform this image
            transformed_moving_img=apply_registration(moving_img, template_img, Rough_Homography_Upscaled, fine_homography_set)
            
            #write the output to the desired directory using bfio
            logger.info('writing output image...')
            transformed_moving_image_5channel=np.zeros((transformed_moving_image.shape[0],transformed_moving_image.shape[1],1,1,1),dtype='uint16')
            transformed_moving_image_5channel[:,:,0,0,0]=transformed_moving_image            
            bw = BioWriter(os.joinpath(outDir,moving_image_name ), metadata=bf.read_metadata() )
            bw.write_image(transformed_moving_image_5channel)
            bw.close_image()             
    
    # Close the javabridge
    logger.info('Closing the javabridge...')
    jutil.kill_vm()
    