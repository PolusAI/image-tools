import cv2
import numpy as np
from bfio.bfio import BioReader, BioWriter
import bioformats     
import javabridge 
import argparse
import logging
import os
import traceback
import math



"""
This script consists of the image registration algorithm. 
The two major functions in this script are (found at the end):
1. register_images()
2. apply_registration()
All the other functions are utility functions. 

register_images() registers the moving image with the template image
and returns the set of transformation(rough and fine) required to do so.
The apply_registration() function takes as an input these set of 
transformations and applies it to an image.
"""


def image_transformation(moving_image,reference_image):    
    '''
    This function registers the moving image with reference image
    Inputs:
        moving_image = Image to be transformed
        reference_image=  reference Image  
    Outputs:
        warped_image = Transformed moving image
        homography= transformation applied to the moving image
      
    '''
    # height, width of the reference image
    height, width = reference_image.shape
    # max number of features to be calculated using ORB 
    max_features=500000   
    # initialize orb feature matcher
    orb = cv2.ORB_create(max_features)
    
    # find keypoints and descriptors in moving and reference image    
    keypoints1, descriptors1 = orb.detectAndCompute(moving_image, None)
    keypoints2, descriptors2 = orb.detectAndCompute(reference_image, None)
    
    # match and sort the descriptos using hamming distance    
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)
    matches.sort(key=lambda x: x.distance, reverse=False)
    
    # extract top 5% of matches
    good_match_percent=0.05    
    numGoodMatches = int(len(matches) * good_match_percent)
    matches = matches[:numGoodMatches]
    
    # extract the point coordinates from the keypoints
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)    
    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt        
    
    # calculate the homography matrix    
    homography, _ = cv2.findHomography(points1, points2, cv2.RANSAC)
    # use the homography matrix to transform the moving image
    warped_img = cv2.warpPerspective(moving_image, homography, (width, height))    
    return warped_img,homography  

def get_scale_factor(height,width):
    """
    This function returns the appropriate scale factor w.r.t to 
    a target size. Target size has been fixed to 5 megapixels.
    Inputs: 
        Image height and width
    Outputs:
            scale factor
    """    
    TARGET_SIZE=5000000 # 5 megapixels
    scale_factor=math.sqrt((height*width)/TARGET_SIZE)    
    return int(scale_factor) if scale_factor>1 else 1

def get_scaled_down_images(image,scale_factor):
    """
    This function returns the scaled down version of an image.    
    Inputs:
        image : input image to be scaled down, (either 8 bit or 16 bit image)
        scale_factor : the factor by which the image needs
                       to be scaled down
    Outputs:
        rescaled_image: 8 bit scaled down version of the input image
    """
    height,width=image.shape
    new_height,new_width=[int(height/scale_factor),int(width/scale_factor)] 
    # opencv and numpy have a different coordinate system   
    rescaled_image=cv2.resize(image,(new_width,new_height))    
    
    # check image dtype and convert to uint8
    if rescaled_image.dtype == 'uint16':    
        rescaled_image=(rescaled_image/256).astype('uint8')    
    elif rescaled_image.dtype == 'uint32':
        rescaled_image=(rescaled_image/2**24).astype('uint8') 
    elif rescaled_image.dtype == 'uint64':
        rescaled_image=rescaled_image=(rescaled_image/2**56).astype('uint8')         
        
    return rescaled_image

def get_tiles(image,buffer):
    """
    This function divides the image into 4 tiles. The number of tiles has been
    fixed to 4. Fixing the number of tiles may leed to memory problems. Following
    versions of this plugin will address this issue. 
    
    Inputs: 
        image: input image to be tiled
        buffer: True or false
        
        If buffer is false then the image is equally divided into 4 tiles. If buffer is true
        then the image is divided into 4 overlapping tiles. The overlapping
        buffer is equal to (height/8,width/8) where height,width=image.shape
        
    Output:
        tiles: a list of 4 tiles 
    """
    
    h,w=image.shape
    
    if not buffer:
        return [image[:int(h/2),:int(w/2)],image[:int(h/2),int(w/2):],
                         image[int(h/2):,:int(w/2)], image[int(h/2):,int(w/2):]]
    else:
        return [image[:int(5*h/8),:int(5*w/8)],image[:int(5*h/8),int(3*w/8):],
                         image[int(3*h/8):,:int(5*w/8)],image[int(3*h/8):,int(3*w/8):]] 
            
    

def apply_rough_homography(image,homography_largescale,reference_image_shape):
    """
    This function transforms the original moving image using the homography matrix.
    In the image_transformation() function above,  cv2.warpPerspective() function has been
    used to apply the homography matrix. cv2.warpPerspective() has a limit on the max image 
    size that it can transform and can't be used to warp the original moving image. 
    
    logic: 
        transformed_image_coordinates= homography * image_coordinates
        homography_inverse * transformed_image = image
    
    Inputs: 
        image : image to be transformed
        homography_largescale: the transformation matrix
        reference_image_shape: the desired output shape of the moving image
        
    output:
        transformed_image : transformed output image
    
    """  
    
    # calculate homography_inverse     
    homography_inverse=np.linalg.inv(homography_largescale)
    
    #dimensions of the reference and moving image
    height_1,width_1=reference_image_shape
    height_2,width_2=image.shape
    
    # initialize the output image matrix
    transformed_image=np.zeros((height_1,width_1),dtype=image.dtype)  
    
    if height_1< 2**16 and width_1 < 2**16:
        dtype = np.uint16
    elif height_1 < 2**32 and width_1 < 2**32:
        dtype = np.uint32
    else:
        dtype = np.uint64

    height_array = np.zeros((1,width_1),dtype=dtype)
    width_array = np.arange(width_1,dtype=dtype)
    depth_array = np.ones((1,width_1),dtype=dtype)   
      
    # iterate over coordinate values row by row
    for i in range(height_1):   
             
        # store homogeneous coordinate([x1,y1,1], [x2,y2,1]...) values in the row array       
        row_array=np.array([width_array,height_array[0],depth_array[0]])
        height_array = height_array + 1 # Do not use += here
        
        #dot product of homography inverse and the row_array(coordinate_array)
        new_array=np.dot(homography_inverse,row_array)
        
        # convert homogeneous coordinates to 2d coordinates
        new_array=np.round(new_array/new_array[2,:], decimals=0)
        new_array=new_array.astype(int)
        
        # remove all coordinates which are out of bounds 
        # (negative coordinates, or that which lie outside the moving image)
        boo = (np.all(new_array>=0, axis =0)) * (new_array[0] <width_2) * (new_array[1] <height_2)
        new_array=new_array[:,boo]
        row_array=row_array[:,boo]
        
        # fetch pixel values from the moving image and place them at their transformed 
        # position in the transformed image        
        # Note: 1) Performing transformation like this can result in multiple pixels in 
        #       the moving image being assigned to the same pixel in the transformed
        #       image. This issue will be resolved in the future versions of this
        #       plugin.        
        #       2) Also, some kind of interpolation is required here. It will be added in 
        #          the following versions of this plugin.               
               
        transformed_image[row_array[1,:], row_array[0,:]]= image[new_array[1,:], new_array[0,:]]    
    return transformed_image

def get_tile_by_tile_transformation(reference_image_tiles, moving_image_tiles, scale_matrix): 
    """
    This function takes as input a set of tiles from the reference image and moving image and
    calculates the homography tranformation between respective tiles. The transformation is
    calculated between scaled down versions of the images, the scale matrix is used to upscale
    the homography matrixes.
    
    Input:
        reference_image_tiles= list of tiles from the reference image
        moving_image_tiles= list of tiles from the moving image
        scale_matrix= used to up_scale the homography matrices 
        
    Output:
        homography_set = list of homography matrixes 
        
    """       
    homography_set=[]       
    for k in range(len(moving_image_tiles)):
        # cacluate transformation betwen corresponding tiles
        _ , homography_matrix=image_transformation(moving_image_tiles[k], reference_image_tiles[k] )
        # upscale and append the matrix
        homography_set.append(homography_matrix*scale_matrix)            
    return homography_set


def apply_registration(moving_image_path,Template_image_shape,Rough_Homography_Upscaled, fine_homography_set):
    
    """
    This function transforms an image using the set of transformations (rough and fine) returned
    by the register_images() function below. 
    
    Inputs: 
        moving_image_path: path to the moving image
        Template_image_shape: dimensions of the reference image
        Rough_Homography_Upscaled: rough homography matrix returned by register_images() function
        fine_homography_set: set of tile by tile transformation matrices returned by register_images(function)
        
    Outputs:
        transformed_moving_image: transformed moving image
        moving_image_metadata: metadata from the moving image       
        
    """
    
    logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger("apply_registration()")
    logger.setLevel(logging.INFO)
    
    # read image using bfio
    bf = BioReader(moving_image_path)
    moving_image_metadata=bf.read_metadata()
    moving_image = bf.read_image(C=[0])    
    moving_image=moving_image[:,:,0,0,0] 
    del bf
            
    height,width=Template_image_shape
    
    # apply rough transformation to the moving image
    rough_transformed_image=apply_rough_homography(moving_image,Rough_Homography_Upscaled,Template_image_shape)
    del moving_image
    # get tiles
    moving_image_tiles=get_tiles(rough_transformed_image, buffer=True)
    # free up memory
    del rough_transformed_image    
 
    # get output tile shapes
    h2,w2=(int(height/2), int(width/2))
    tile_shapes=[(w2,h2),(width-w2,h2), (w2,height-h2), (width-w2,height-h2)]
    # apply tile by tile transformation to the tiles    
    transformed_moving_image_tiles=[]
    for k in range(len(moving_image_tiles)):
        transformed_moving_image_tiles.append( cv2.warpPerspective(moving_image_tiles[k],fine_homography_set[k],tile_shapes[k]))
    
    # free up memory   
    del moving_image_tiles         
    # stack tiles to get desired output
    logger.info("stack transformed tiles...")   
    
    upperhalf= np.hstack((transformed_moving_image_tiles[0],transformed_moving_image_tiles[1]))
    transformed_moving_image_tiles.pop(0)
    transformed_moving_image_tiles.pop(0)
    lowerhalf=np.hstack((transformed_moving_image_tiles[0],transformed_moving_image_tiles[1]))
    del transformed_moving_image_tiles     
    transformed_moving_image=np.vstack((upperhalf,lowerhalf))    
    del upperhalf
    del lowerhalf     
    logger.info("moving image registered...")  

    return transformed_moving_image , moving_image_metadata
    
    
def register_images(reference_image_path, moving_image_path):
    """
    This function registers the moving image with the reference image 
    and returns the set of transformations required to do so. At first 
    the moving image undergoes a transformation to roughly align it with
    the template image. Following that, the roughly transformed moving
    image is tranformed tile-by-tile to get a finer transformation.
    
    Input:
        reference_image_path: path to the reference image
        moving_image_path: path to the moving image
        
     output:
        transformed_moving_image: output image
        Rough_Homography_Upscaled: matrix used for rough transformation
        fine_homography_set: set of tile by tle tranformations used for fine transformation
        reference_image_shape: dimensions of the reference image
        moving_image_metadata: moving image metadata
            
    """
     
    # initialize logger    
    logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger("register_images()")
    logger.setLevel(logging.INFO)
    
    # read reference image
    bf = BioReader(reference_image_path)
    reference_image = bf.read_image(C=[0])
    reference_image=reference_image[:,:,0,0,0]
    
    reference_image_shape= reference_image.shape   
    height,width=reference_image_shape
    # intialize the scale factor and scale matrix(to be used to upscale the transformation matrices)
    scale_factor=get_scale_factor(reference_image_shape[0],reference_image_shape[1])
    scale_matrix = np.array([[1,1,scale_factor],[1,1,scale_factor],[1/scale_factor,1/scale_factor,1]])
    
    # downscale the reference image
    reference_image_downscaled= get_scaled_down_images(reference_image,scale_factor)
    
    # free up memory
    del reference_image
    
    # read moving image
    bf = BioReader(moving_image_path)
    moving_image_metadata=bf.read_metadata()
    moving_image = bf.read_image(C=[0])
    moving_image=moving_image[:,:,0,0,0]
    del bf
    
    # downscale the moving image
    moving_image_downscaled= get_scaled_down_images(moving_image,scale_factor)
    logger.info("calculating rough homography...")
    
    # calculate rough transformation between scaled down reference and moving image
    _,Rough_Homography_Downscaled = image_transformation(moving_image_downscaled,reference_image_downscaled)
    
    del moving_image_downscaled
    # upscale the rough homography matrix
    Rough_Homography_Upscaled=Rough_Homography_Downscaled*scale_matrix    
    
    # apply upscale rough homography to original moving image
    logger.info("applying rough homography to the moving image...")   
  
    moving_image_transformed=apply_rough_homography(moving_image,Rough_Homography_Upscaled,reference_image_shape)     
    
    # free up memory
    del moving_image          
    
    # downscale the transformed moving image
    moving_image_transformed_downscaled=get_scaled_down_images(moving_image_transformed,scale_factor)
    # downscaled reference image tiles without buffer
    reference_image_tiles=get_tiles(reference_image_downscaled, buffer=False)
    del reference_image_downscaled      
    # downscaled moving image tiles with buffer
    moving_image_tiles=get_tiles(moving_image_transformed_downscaled, buffer=True)
    del moving_image_transformed_downscaled
    # get title by tile upscaled transformation matrices
    logger.info("get tile by tile transformation...")     
    fine_homography_set=get_tile_by_tile_transformation(reference_image_tiles,moving_image_tiles, scale_matrix ) 
    del reference_image_tiles
    del moving_image_tiles    
    # get tiles from original size moving image                                                                               
    moving_image_tiles=get_tiles(moving_image_transformed, buffer=True)  
    del moving_image_transformed 
    
    # get output tile shapes
    h2,w2=(int(height/2), int(width/2))
    tile_shapes=[(w2,h2),(width-w2,h2), (w2,height-h2), (width-w2,height-h2)]    
    # apply tile by tile transformation to the rough transformed moving image  
    transformed_moving_image_tiles=[]
    for k in range(4):
        transformed_moving_image_tiles.append(cv2.warpPerspective(moving_image_tiles[k],fine_homography_set[k],tile_shapes[k])) 
    # free up memory   
    del moving_image_tiles         
    # stack tiles to get desired output
    logger.info("stack transformed tiles...")   
    
    upperhalf= np.hstack((transformed_moving_image_tiles[0],transformed_moving_image_tiles[1]))
    transformed_moving_image_tiles.pop(0)
    transformed_moving_image_tiles.pop(0)
    lowerhalf=np.hstack((transformed_moving_image_tiles[0],transformed_moving_image_tiles[1]))
    del transformed_moving_image_tiles 
    
    transformed_moving_image=np.vstack((upperhalf,lowerhalf))    
    del upperhalf
    del lowerhalf     
    logger.info("moving image registered...")  
    
    return transformed_moving_image, Rough_Homography_Upscaled, fine_homography_set, reference_image_shape,moving_image_metadata
##################################################################################################################################################################

if __name__=="__main__":
    
    try:
        # Initialize the logger
        logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                            datefmt='%d-%b-%y %H:%M:%S')
        logger = logging.getLogger("subprocess-image_registration.py")
        logger.setLevel(logging.INFO)

        # Setup the argument parsing
        logger.info("Parsing arguments...")
        parser = argparse.ArgumentParser(prog='imageRegistration', description='This script registers an image collection')
        parser.add_argument('--registrationString', dest='registration_string', type=str, required=True)
        parser.add_argument('--similarTransformationString', dest='similar_transformation_string', type=str, required=True)
        parser.add_argument('--outDir', dest='outDir', type=str, required=True)
        parser.add_argument('--template', dest='template', type=str,  required=True)

        # parse the arguments 
        args = parser.parse_args()
        registration_string = args.registration_string
        logger.info('registration_string = {}'.format(registration_string))
        similar_transformation_string = args.similar_transformation_string
        logger.info('similar_transformation_string = {}'.format(similar_transformation_string))
        outDir = args.outDir   
        template = args.template
        
        logger.info('starting javabridge...')
        javabridge.start_vm(class_path=bioformats.JARS)
        
        # extract filenames from registration_string and similar_transformation_string
        registration_set=registration_string.split()
        similar_transformation_set=similar_transformation_string.split()
        
        del registration_string
        del similar_transformation_string
        
        filename_len=len(template)
        
        logger.info('calculating transformation to register image {} to {}'.format(registration_set[1][-1*filename_len:] ,registration_set[0][-1*filename_len:]))    
        # seperate the filename of the moving image from the complete path
        moving_image_name=registration_set[1][-1*filename_len:]        
        
        # register the images and store the set of transformation used to carry out the registration
        transformed_moving_image, Rough_Homography_Upscaled, fine_homography_set, template_img_shape,moving_image_metadata=register_images(registration_set[0], registration_set[1])
        
        logger.info('transformation calculated and creating 5 channel output of {}'.format(moving_image_name))
        # write the output to the desired directory using bfio
        transformed_moving_image_5channel=np.zeros((transformed_moving_image.shape[0],transformed_moving_image.shape[1],1,1,1),dtype='uint16')
        transformed_moving_image_5channel[:,:,0,0,0]=transformed_moving_image 
        
        
        del transformed_moving_image  
        logger.info("writing output using bfio...")        
        bw = BioWriter(os.path.join(outDir,moving_image_name), metadata=moving_image_metadata)
        bw.num_x(transformed_moving_image_5channel.shape[1])
        bw.num_y(transformed_moving_image_5channel.shape[0])
        bw.num_z(1)
        bw.num_c(1)
        bw.num_t(1)
        bw.write_image(transformed_moving_image_5channel)
        bw.close_image() 
        del bw       
            
        del transformed_moving_image_5channel     
        
        # iterate across all images which have the similar transformation as the moving image above  
        logger.info('iterate over all images that have similar transformation')
        for moving_image_path in similar_transformation_set:
            
            # seperate image name from the path to it
            moving_image_name=moving_image_path[-1*filename_len:]            
            
            logger.info('applying transformation to image {}'.format(moving_image_name))
                        
            # use the precalculated transformation to transform this image
            transformed_moving_img, moving_image_metadata =apply_registration(moving_image_path, template_img_shape, Rough_Homography_Upscaled, fine_homography_set)          

            #write the output to the desired directory using bfio
            logger.info('writing output image...')
            transformed_moving_image_5channel=np.zeros((transformed_moving_img.shape[0],transformed_moving_img.shape[1],1,1,1),dtype='uint16')
            transformed_moving_image_5channel[:,:,0,0,0]=transformed_moving_img       

            del transformed_moving_img             
            bw = BioWriter(os.path.join(outDir,moving_image_name ), metadata=moving_image_metadata)
            bw.num_x(transformed_moving_image_5channel.shape[1])
            bw.num_y(transformed_moving_image_5channel.shape[0])
            bw.num_z(1)
            bw.num_c(1)
            bw.num_t(1)
            bw.write_image(transformed_moving_image_5channel)
            bw.close_image()         
            del bw  
            del transformed_moving_image_5channel  
            
    except Exception:
        traceback.print_exc() 
    
    finally:    
        # Close the javabridge
        logger.info('Closing the javabridge...')
        javabridge.kill_vm()   
    
    
    
    
    
   