import cv2
import numpy as np
from bfio.bfio import BioReader, BioWriter
import bioformats     
import javabridge 
import argparse
import logging
import traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count

def corr2(a,b):
    """corr2 Calculate correlation between 2 images

    Inputs:
        a (np.ndarray): A 2-dimensional numpy array
        b (np.ndarray): A 2-dimensional numpy array

    Outputs:
        float: the correlation between a and b
    """
    
    c = np.sum(a)/np.size(a)
    d = np.sum(b)/np.size(b)
    
    c = a - c
    d = b - d
    
    r = (c*d).sum() / np.sqrt((c*c).sum() * (d*d).sum())
    
    return r

def get_transform(moving_image,reference_image,max_val,min_val):
    """get_transform Calculate homography matrix transform

    This function registers the moving image with reference image
    
    Inputs:
        moving_image = Image to be transformed
        reference_image=  reference Image  
    Outputs:
        homography= transformation applied to the moving image
    """
    # height, width of the reference image
    height, width = reference_image.shape
    # max number of features to be calculated using ORB
    max_features=500000  
    # initialize orb feature matcher
    orb = cv2.ORB_create(max_features)
    
    # Normalize images and convert to appropriate type
    moving_image_norm = cv2.GaussianBlur(moving_image,(3,3),0)
    reference_image_norm = cv2.GaussianBlur(reference_image,(3,3),0)
    moving_image_norm = (moving_image_norm-min_val)/(max_val-min_val)
    moving_image_norm = (moving_image_norm * 255).astype(np.uint8)
    reference_image_norm = (reference_image_norm-min_val)/(max_val-min_val)
    reference_image_norm = (reference_image_norm * 255).astype(np.uint8)
    
    # find keypoints and descriptors in moving and reference image
    keypoints1, descriptors1 = orb.detectAndCompute(moving_image_norm, None)
    keypoints2, descriptors2 = orb.detectAndCompute(reference_image_norm, None)
    
    # Escape if one image does not have descriptors
    if not (isinstance(descriptors1,np.ndarray) and isinstance(descriptors2,np.ndarray)):
        return None
    
    # match and sort the descriptos using hamming distance    
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)
    matches.sort(key=lambda x: x.distance, reverse=False)
    
    # extract top 25% of matches
    good_match_percent=0.25    
    numGoodMatches = int(len(matches) * good_match_percent)
    matches = matches[:numGoodMatches]
    
    # extract the point coordinates from the keypoints
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)    
    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt        
    
    # If no matching points, return None
    if points1.shape[0]==0 or points2.shape[0]==0:
        return None
    
    # calculate the homography matrix
    homography, _ = cv2.findHomography(points1, points2, cv2.RANSAC)
    
    return homography

def get_scale_factor(height,width):
    """
    This function returns the appropriate scale factor w.r.t to 
    a target size. Target size has been fixed to 5 megapixels.
    
    Inputs:
        height (int): Image height
        width (int): Image width
    Outputs:
        scale factor
    """    
    TARGET_SIZE=5000000 # 5 megapixels
    scale_factor=np.sqrt((height*width)/TARGET_SIZE)    
    return int(scale_factor) if scale_factor>1 else 1

def get_scaled_down_images(image,scale_factor,get_max=False):
    """
    This function returns the scaled down version of an image.    
    Inputs:
        image : A BioReader object
        scale_factor : the factor by which the image needs
                       to be scaled down
    Outputs:
        rescaled_image: scaled down version of the input image
    """
    
    # Calculate scaling variables
    stride = int(scale_factor * np.floor(4096/scale_factor))
    width = np.ceil(image.num_y()/scale_factor).astype(int)
    height = np.ceil(image.num_x()/scale_factor).astype(int)
    
    # Initialize the output
    rescaled_image = np.zeros((width,height),dtype=image._pix['type'])
    
    # If max value is requested, initialize the variables
    if get_max:
        max_val = 0
        min_val = np.inf
        
    def load_and_scale(X,Y,x,y,get_max=get_max,reader=image,scale_factor=scale_factor,rescaled_image=rescaled_image):
        """load_and_scale Load a section of an image and downscale
        
        This is a transient method, and only works within the get
        scaled_down_images method.
        It's used to thread out loading and downscaling of large images.
        
        """
        # Attach the JVM to this thread
        javabridge.attach()
        
        # Read an image tile
        tile = reader.read_image(X=X,Y=Y,Z=[0,1],C=[0],T=[0]).squeeze()
        
        javabridge.detach()
        
        # Average the image for scaling
        blurred_image = cv2.boxFilter(tile,-1,(scale_factor,scale_factor))
        
        # Collect pixels for downscaled image
        rescaled_image[y[0]:y[1],x[0]:x[1]] = blurred_image[::scale_factor,::scale_factor]
        
        if get_max:
            return np.max(tile),np.min(tile)
        else:
            return None
    
    # Load and downscale the image
    threads = []
    workers = max([cpu_count()-1,1])
    with ThreadPoolExecutor(workers) as executor:
        for x in range(0,image.num_x(),stride):
            x_max = np.min([x+stride,image.num_x()]) # max x to load
            xi = int(x//scale_factor)                # initial scaled x-index
            xe = int(np.ceil(x_max/scale_factor))    # ending scaled x-index
            for y in range(0,image.num_y(),stride):
                y_max = np.min([y+stride,image.num_y()]) # max y to load
                yi = int(y//scale_factor)                # initial scaled y-index
                ye = int(np.ceil(y_max/scale_factor))    # ending scaled y-index
                
                threads.append(executor.submit(load_and_scale,[x,x_max],[y,y_max],[xi,xe],[yi,ye]))
    
    # Return max and min values if requested
    if get_max:
        results = [thread.result() for thread in threads]
        max_val = max(result[0] for result in results)
        min_val = max(result[1] for result in results)
        return rescaled_image,max_val,min_val
    else:
        return rescaled_image

def register_image(br_ref,br_mov,bw,Xt,Yt,Xm,Ym,x,y,X_crop,Y_crop,max_val,min_val):
    """register_image Register one section of two images

    This method is designed to be used within a thread. It registers
    one section of two different images, saves the output, and
    returns the homography matrix used to transform the image.

    """
    # Attach the JVM to this thread
    javabridge.attach()
    
    # Load a section of the reference and moving images
    ref_tile = br_ref.read_image(X=[Xt[0],Xt[1]],Y=[Yt[0],Yt[1]],Z=[0,1],C=[0],T=[0]).squeeze()
    mov_tile = br_mov.read_image(X=[Xm[0],Xm[1]],Y=[Ym[0],Ym[1]],Z=[0,1],C=[0],T=[0]).squeeze()
    
    # Get the transformation matrix
    projective_transform = get_transform(mov_tile,ref_tile,max_val,min_val)
    
    # Use the rough transformation matrix if no matrix was returned
    is_rough = False
    if not isinstance(projective_transform,np.ndarray):
        is_rough = True
        projective_transform = Rough_Homography_Upscaled
    
    # Transform the moving image
    transformed_image = cv2.warpPerspective(mov_tile,projective_transform,(Xt[1]-Xt[0],Yt[1]-Yt[0]))
    
    # Determine the correlation between the reference and transformed moving image
    corr = corr2(ref_tile,transformed_image)
    
    # If the correlation is bad, try using the rough transform instead
    if corr < 0.4 and not is_rough:
        rough_image = cv2.warpPerspective(mov_tile,Rough_Homography_Upscaled,(Xt[1]-Xt[0],Yt[1]-Yt[0]))
        corr_rough = corr2(ref_tile,rough_image)
        if corr_rough > corr:
            projective_transform = Rough_Homography_Upscaled
            transformed_image = rough_image
    
    # Write the transformed moving image
    bw.write_image(transformed_image[Y_crop[0]:Y_crop[1],X_crop[0]:X_crop[1],np.newaxis,np.newaxis,np.newaxis],X=[x],Y=[y])
    
    # Detach the JVM from this thread
    javabridge.detach()
    
    return projective_transform

def apply_transform(br_mov,bw,tiles,shape,transform):
    """apply_transform Apply a transform to an image

    This method is designed to be used within a thread. It loads
    a section of an image, applies a transform, and saves the
    transformed image to file.

    """
    # Attach the JVM to the thread
    javabridge.attach()
    
    # Get the tile indices
    Xm,Ym,Xt,Yt = tiles
    
    # Read the moving image tile
    mov_tile = br_mov.read_image(X=[Xm[0],Xm[1]],Y=[Ym[0],Ym[1]],Z=[0,1],C=[0],T=[0]).squeeze()
    
    # Get the image coordinates and shape
    x,y,X_crop,Y_crop = shape
    
    # Transform the image
    transformed_image = cv2.warpPerspective(mov_tile,transform,(Xt[1]-Xt[0],Yt[1]-Yt[0]))
    
    # Write the transformed image to the output file
    bw.write_image(transformed_image[Y_crop[0]:Y_crop[1],X_crop[0]:X_crop[1],np.newaxis,np.newaxis,np.newaxis],[x],[y])
    
    # Detach the JVM from the thread
    javabridge.detach()

if __name__=="__main__":
    
    try:
        # Initialize the logger
        logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                            datefmt='%d-%b-%y %H:%M:%S')
        logger = logging.getLogger("image_registration.py")
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
        similar_transformation_string = args.similar_transformation_string
        outDir = args.outDir   
        template = args.template
        
        logger.info('starting javabridge...')
        # Initialize log4j to keep it quiet
        log_config = Path(__file__).parent.joinpath("log4j.properties")
        # Start javabridge
        javabridge.start_vm(args=["-Dlog4j.configuration=file:{}".format(str(log_config.absolute()))],class_path=bioformats.JARS)
        
        # extract filenames from registration_string and similar_transformation_string
        registration_set=registration_string.split()
        similar_transformation_set=similar_transformation_string.split()
        print(similar_transformation_set)
        
        filename_len=len(template)
          
        # seperate the filename of the moving image from the complete path
        moving_image_name=registration_set[1][-1*filename_len:]
        
        # read and downscale reference image
        br_ref = BioReader(registration_set[0])
        scale_factor=get_scale_factor(br_ref.num_y(),br_ref.num_x())
        logger.info('Scale factor: {}'.format(scale_factor))
        
        # intialize the scale factor and scale matrix(to be used to upscale the transformation matrices)
        scale_matrix = np.array([[1,1,scale_factor],[1,1,scale_factor],[1/scale_factor,1/scale_factor,1]])
        
        logger.info('Reading and downscaling reference image: {}'.format(Path(registration_set[0]).name))
        reference_image_downscaled,max_val,min_val = get_scaled_down_images(br_ref,scale_factor,get_max=True)
        
        # read moving image
        logger.info('Reading and downscaling moving image: {}'.format(Path(registration_set[1]).name))
        br_mov = BioReader(registration_set[1])
        moving_image_downscaled = get_scaled_down_images(br_mov,scale_factor)
        
        # calculate rough transformation between scaled down reference and moving image
        logger.info("calculating rough homography...")
        Rough_Homography_Downscaled = get_transform(moving_image_downscaled,
                                                    reference_image_downscaled,
                                                    max_val,
                                                    min_val)
        
        # upscale the rough homography matrix
        Rough_Homography_Upscaled=Rough_Homography_Downscaled*scale_matrix
        homography_inverse=np.linalg.inv(Rough_Homography_Upscaled)
        
        # Initialize the output file
        bw = BioWriter(str(Path(outDir).joinpath(Path(registration_set[1]).name)),metadata=br_mov.read_metadata())
        bw.num_x(br_ref.num_x())
        bw.num_y(br_ref.num_y())
        bw.num_z(1)
        bw.num_c(1)
        bw.num_t(1)
        
        # transformation variables
        reg_shape = []
        reg_tiles = []
        reg_homography = []
        
        # thread list
        threads = []
        
        # First tile flag, explained below
        first_tile = True
        
        # Loop through image tiles and start threads
        with ThreadPoolExecutor(max([3*cpu_count()//4,1])) as executor:
            for x in range(0,br_ref.num_x(),2048):
                for y in range(0,br_ref.num_y(),2048):
            
                    # Get reference/template image coordinates
                    Xt = [np.max([0,x-1024]),np.min([br_ref.num_x(),x+2048+1024])]
                    Yt = [np.max([0,y-1024]),np.min([br_ref.num_y(),y+2048+1024])]
                    
                    # Use the rough homography to get coordinates in the moving image
                    coords = np.array([[Xt[0],Xt[0],Xt[1],Xt[1]],
                                    [Yt[0],Yt[1],Yt[1],Yt[0]],
                                    [1,1,1,1]],
                                    dtype=np.float64)
                    
                    coords = np.matmul(homography_inverse,coords)
                    
                    mins = np.min(coords,axis=1)
                    maxs = np.max(coords,axis=1)
                    
                    Xm = [int(np.floor(np.max([mins[0],0]))),
                          int(np.ceil(np.min([maxs[0],br_mov.num_x()])))]
                    Ym = [int(np.floor(np.max([mins[1],0]))),
                          int(np.ceil(np.min([maxs[1],br_mov.num_y()])))]
                    
                    reg_tiles.append((Xm,Ym,Xt,Yt))
                    
                    # Get cropping dimensions
                    X_crop = [1024 if Xt[0] > 0 else 0]
                    X_crop.append(2048+X_crop[0] if Xt[1]-Xt[0] >= 3072 else Xt[1]-Xt[0]+X_crop[0])
                    Y_crop = [1024 if Yt[0] > 0 else 0]
                    Y_crop.append(2048+Y_crop[0] if Yt[1]-Yt[0] >= 3072 else Yt[1]-Yt[0]+Y_crop[0])
                    reg_shape.append((x,y,X_crop,Y_crop))
                    
                    # Start a thread to register the tiles
                    threads.append(executor.submit(register_image,br_ref,br_mov,bw,Xt,Yt,Xm,Ym,x,y,X_crop,Y_crop,max_val,min_val))
                    
                    # Bioformats require the first tile be written before any other tile
                    if first_tile:
                        first_tile = False
                        threads[0].result()
        
            # Wait for threads to finish, track progress
            for thread_num in range(len(threads)):
                if thread_num % 10 == 0:
                    logger.info('Registration progress: {:6.2f}%'.format(100*thread_num/len(threads)))
                reg_homography.append(threads[thread_num].result())
                
        # Close the image
        bw.close_image()
        logger.info('Registration progress: {:6.2f}%'.format(100.0))
        
        # iterate across all images which have the similar transformation as the moving image above
        for moving_image_path in similar_transformation_set:
            
            # seperate image name from the path to it
            moving_image_name=moving_image_path[-1*filename_len:]
            
            logger.info('Applying registration to image: {}'.format(moving_image_name))
            
            br_mov = BioReader(moving_image_path)
     
            bw = BioWriter(str(Path(outDir).joinpath(moving_image_name)), metadata=br_mov.read_metadata())
            bw.num_x(br_ref.num_x())
            bw.num_y(br_ref.num_y())
            bw.num_z(1)
            bw.num_c(1)
            bw.num_t(1)
            
            # Apply transformation to remaining images
            logger.info('Transformation progress: {:5.2f}%'.format(0.0))
            threads = []
            with ThreadPoolExecutor(max([cpu_count()-1,1])) as executor:
                first_tile = True
                for tile,shape,transform in zip(reg_tiles,reg_shape,reg_homography):
                
                    # Start transformation threads
                    threads.append(executor.submit(apply_transform,br_mov,bw,tile,shape,transform))
                    
                    # The first tile must be written before all other tiles
                    if first_tile:
                        first_tile = False
                        threads[0].result()

                # Wait for threads to finish and track progress 
                for thread_num in range(len(threads)):
                    if thread_num % 10 == 0:
                        logger.info('Transformation progress: {:6.2f}%'.format(100*thread_num/len(threads)))
                    threads[thread_num].result()
            logger.info('Transformation progress: {:6.2f}%'.format(100.0))
            
            bw.close_image()
            
    except Exception:
        traceback.print_exc() 
    
    finally:    
        # Close the javabridge
        logger.info('Closing the javabridge...')
        javabridge.kill_vm()
   