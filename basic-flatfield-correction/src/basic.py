import argparse, time, csv, logging, re, copy, cv2, bioformats

import numpy as np
import javabridge as jutil

from bfio.bfio import BioReader,BioWriter
from pathlib import Path
from utils import _parse_files, _parse_fpattern, _get_output_name

# Disable java logging
def init_logger(self):
    pass
bioformats.init_logger = init_logger

OPTIONS = {'max_iterations': 500,
           'max_reweight_iterations': 10,
           'optimization_tol': 10**-6,
           'reweight_tol': 10**-4,
           'darkfield': False,
           'size': 128,
           'epsilon': 0.1
          }

""" Load files and create an image stack """
def _get_resized_image_stack(fpath,flist):
    """ Load all images in a list and resize to OPTIONS['size']
    
    When files are parsed, the variables are used in an index to provide
    a method to reference a specific file name by its dimensions. This
    function returns the variable index based on the input filename pattern.

    Inputs:ed th
        fpath - Location of images
        flist - List of images to load and resize
    Outputs:
        img_stack - A 3D stack of 2D images
        X - width of image
        Y - height of image
    """
    
    #Initialize the output
    br = BioReader(str(Path(fpath).joinpath(flist[0])))
    X = br.num_x()
    Y = br.num_y()
    C = len(flist)
    img_stack = np.zeros((OPTIONS['size'],OPTIONS['size'],C),dtype=np.float32)
    
    # Load every image as a z-slice
    for ind,fname in zip(range(len(flist)),flist):
       br = BioReader(str(Path(fpath).joinpath(fname)))
       I = np.squeeze(br.read_image())
       img_stack[:,:,ind] = cv2.resize(I,(OPTIONS['size'],OPTIONS['size']),interpolation=cv2.INTER_LINEAR).astype(np.float32)
       
    return img_stack,X,Y

def _dct2(D):
    """ Discrete cosine transform
    
    This function originally existed to replicate Matlabs dct function using scipy's dct function. It was necessary
    to perform the dct along the rows and columns. To simplify the container creation, the dct function from opencv
    was used. This function is slower, and scipy's version should be implemented in the future.

    The scipy dct that was originally used was type-II.

    Inputs:
        D - Input matrix to perform dct on
    Outputs:
        d - The dct of input, D
    """
    d = cv2.dct(D).astype(np.float64)
    return d

def _idct2(D):
    """ Discrete cosine transform
    
    This function originally existed to replicate Matlabs dct function using scipy's dct function. It was necessary
    to perform the dct along the rows and columns. To simplify the container creation, the dct function from opencv
    was used. This function is slower, and scipy's version should be implemented in the future.

    The scipy dct that was originally used was type-III, which is the inverse of dct type-II

    Inputs:
        D - Input matrix to perform dct on
    Outputs:
        d - The dct of input, D
    """
    d = cv2.dct(D,flags=cv2.DCT_INVERSE).astype(np.float64)
    return d

def _initialize_options(img_stack,get_darkfield,options):
    """ Initialize optimization options
    
    This function modifies the default OPTIONS using information about the images to be processed.

    Inputs:
        img_stack - A numpy matrix where images are concatenated along the 3rd dimensions
        get_darkfield - If true, estimate the darkfield image
        options - An existing set of options to be modified
    Outputs:
        new_options - Modified options
    """
    meanD = np.mean(img_stack,2)
    meanD = meanD/np.mean(meanD)
    weights = _dct2(meanD)
    new_options = copy.deepcopy(options)
    new_options['lambda'] = np.sum(np.abs(weights))/800
    new_options['lambda_darkfield'] = np.sum(np.abs(weights))/2000
    new_options['weight'] = np.ones(img_stack.shape,dtype=np.float32)
    new_options['darkfield_limit'] = 10**7
    new_options['darkfield'] = get_darkfield
    return new_options

def _inexact_alm_l1(D,options):
    """ L1 optimization using inexact augmented Legrangian multipliers (IALM)
    
    This function finds the smallest number of features in Fourier space (frequencies) that minimizes the
    error when compared to images, using L1 loss and IALM optimization. By using L1 loss in Fourier space,
    the resulting background flatfield image should be an image with smooth gradients.

    This function is based off of the Matlab version of BaSiC found here:
    https://github.com/QSCD/BaSiC/blob/master/inexact_alm_rspca_l1.m

    Additional information on  IALM can be found on arXiv:
    https://arxiv.org/pdf/1009.5055.pdf

    Inputs:
        D - Stack of images used to estimate the flatfield
        options - optimization options
    Outputs:
        new_options - Modified options
    """
    # Get basic image information and reshape input
    p = D.shape[0]
    q = D.shape[1]
    m = p*q
    n = D.shape[2]
    D = np.reshape(D,(m,n))
    options['weight'] = np.reshape(options['weight'],D.shape)
    
    # Matrix normalization factor
    temp = np.linalg.svd(D,full_matrices=False,compute_uv=False)
    norm_two = np.float64(temp[0])
    del temp
    
    # A is a low rank matrix that is being solved for
    A = np.zeros(D.shape,dtype=np.float32)
    A_coeff = np.ones((1,n),dtype=np.float64)   # per image scaling coefficient, accounts for things like photobleaching
    A_offset = np.zeros((m,1),dtype=np.float64) # offset per pixel across all images

    # E1 is the additive error. Since the goal is determining the background signal, this is the real signal at each pixel
    E1 = np.zeros(D.shape,dtype=np.float32)

    # Normalization factors
    ent1 = np.float64(1)    # flatfield normalization
    ent2 = np.float64(10)   # darkfield normalization

    # Weights
    W = _dct2(np.mean(np.reshape(A,(p,q,n)),2))

    # Initialize gradient and weight normalization factors
    Y1 = np.float64(0)          
    mu = np.float64(12.5)/norm_two
    mu_bar = mu * 10**7
    rho = np.float64(1.5)

    # Frobenius norm
    d_norm = np.linalg.norm(D,'fro')

    # Darkfield upper limit and offset
    B1_uplimit = np.min(D)
    B1_offset = np.float64(0)
    
    # Perform optimization
    iternum = 0
    converged = False
    while not converged:
        iternum += 1

        # Calculate the flatfield using existing weights, coefficients, and offsets
        W_idct_hat = _idct2(W)
        A = np.matmul(np.reshape(W_idct_hat,(m,1)),A_coeff) + A_offset
        temp_W = np.divide(D - A - E1 + np.multiply(1/mu,Y1),ent1)
        
        # Update the weights
        temp_W = np.reshape(temp_W,(p,q,n))
        temp_W = np.mean(temp_W,2)
        W = W + _dct2(temp_W)
        W = np.max(np.reshape(W - options['lambda']/(ent1*mu),(p,q,1)),-1,initial=0) + np.min(np.reshape(W + options['lambda']/(ent1*mu),(p,q,1)),-1,initial=0)
        W_idct_hat = _idct2(W)
        
        # Calculate the flatfield using updated weights
        A = np.matmul(np.reshape(W_idct_hat,(m,1)),A_coeff) + A_offset
        
        # Determine the error
        E1 = E1 + np.divide(D - A - E1 + np.multiply(1/mu,Y1),ent1)
        E1 = np.max(np.reshape(E1 - options['weight']/(ent1*mu),(m,n,1)),-1,initial=0) + np.min(np.reshape(E1 + options['weight']/(ent1*mu),(m,n,1)),-1,initial=0)
        
        # Calculate the flatfield coefficients by subtracting the errors from the original data
        R1 = D-E1
        A_coeff = np.reshape(np.mean(R1,0)/np.mean(R1),(1,n))
        A_coeff[A_coeff<0] = 0       # pixel values should never be negative
        
        # Calculate the darkfield component if specified by the user
        if options['darkfield']:
            # Get images with predominantly background pixels
            validA1coeff_idx = np.argwhere(A_coeff<1)[:,1]
            R1_upper = R1[np.argwhere(np.reshape(W_idct_hat,(-1,1)).astype(np.float64)>(np.float64(np.mean(W_idct_hat))-np.float64(10**-5)))[:,0],:]
            R1_upper = np.mean(R1_upper[:,validA1coeff_idx],0)
            R1_lower = R1[np.argwhere(np.reshape(W_idct_hat,(-1,1))<np.mean(W_idct_hat)+np.float64(10**-5))[:,0],:]
            R1_lower = np.mean(R1_lower[:,validA1coeff_idx],0)
            B1_coeff = (R1_upper-R1_lower)/np.mean(R1)
            k = validA1coeff_idx.size
            
            # Calculate the darkfield offset
            temp1 = np.sum(np.square(A_coeff[0,validA1coeff_idx]))
            temp2 = np.sum(A_coeff[0,validA1coeff_idx])
            temp3 = np.sum(B1_coeff)
            temp4 = np.sum(A_coeff[0,validA1coeff_idx]*B1_coeff)
            temp5 = temp2 * temp3 - k*temp4
            if temp5 == 0:
                B1_offset = np.float64(0)
            else:
                B1_offset = (temp1*temp3-temp2*temp4)/temp5
            B1_offset = np.max(B1_offset,initial=0)
            B1_offset = np.min(B1_offset,initial=B1_uplimit/(np.mean(W_idct_hat)+10**-7))
            B_offset = B1_offset * np.mean(W_idct_hat) - B1_offset*np.reshape(W_idct_hat,(-1,1))
            
            # Calculate darkfield
            A1_offset = np.reshape(np.mean(R1[:,validA1coeff_idx],1),(-1,1)) - np.mean(A_coeff[0,validA1coeff_idx]) * np.reshape(W_idct_hat,(-1,1))
            A1_offset = A1_offset - np.mean(A1_offset)
            A_offset = A1_offset - np.mean(A1_offset) - B_offset
            
            # Update darkfield weights
            W_offset = _dct2(np.reshape(A_offset,(p,q)))
            W_offset = np.max(np.reshape(W_offset - options['lambda_darkfield']/(ent2*mu),(p,q,1)),-1,initial=0) \
                     + np.min(np.reshape(W_offset + options['lambda_darkfield']/(ent2*mu),(p,q,1)),-1,initial=0)

            # Calculate darkfield based on updated weights
            A_offset = _idct2(W_offset)
            A_offset = np.reshape(A_offset,(-1,1))
            A_offset = np.max(np.reshape(A_offset - options['lambda_darkfield']/(ent2*mu),(A_offset.shape[0],A_offset.shape[1],1)),-1,initial=0) \
                     + np.min(np.reshape(A_offset + options['lambda_darkfield']/(ent2*mu),(A_offset.shape[0],A_offset.shape[1],1)),-1,initial=0)
            A_offset = A_offset + B_offset
            
        # Loss
        Z1 = D - A - E1
        
        # Update weight regularization term
        Y1 = Y1 + mu*Z1
        
        # Update learning rate
        mu = np.min(mu*rho,initial=mu_bar)
        
        # Stop if loss is below threshold
        stopCriterion = np.linalg.norm(Z1,ord='fro')/d_norm
        if stopCriterion < options['optimization_tol'] or iternum > options['max_iterations']:
            converged = True
        
    # Calculate final darkfield image
    A_offset = A_offset + B1_offset * np.reshape(W_idct_hat,(-1,1))
    
    return A,E1,A_offset


def _get_flatfield_and_reweight(X_k_A,X_k_E,X_k_Aoffset,options):
    """ Format flatfield/darkfield and change weights
    
    The inexact augmented legrangian multiplier method uses L1 loss, but this is only done
    since an exact solution to L0 problems do not exist. After each round of optimization, the
    starting weights are recalculated to give low weights to pixels that are not background and
    high weights to background pixels. Then another round of optimization is performed.

    Since the value of the weights are tied to the values of the flatfield and darkfield values,
    this function formats both the flatfield and darkfield images for output and returns an updated
    weight matrix.

    The flatfield image is normalized so that the mean pixel value is 1. The darkfield image
    contains raw pixel values.

    Inputs:
        D - Numpy stack of images
        flatfield - numpy floating precision matrix containing flatfield values
        darkfield - numpy floating precision matrix containing darkfield values
    Outputs:
        flatfield - A floating precision numpy matrix containing normalized flatfield values
        darkfield - A floating precision numpy matrix containing darkfield pixel values
        options - optimization options with new weights (this is passed only for code readability)
    """
    XA = np.reshape(X_k_A,(options['size'],options['size'],-1))
    XE = np.reshape(X_k_E,(options['size'],options['size'],-1))
    XE_norm = XE/np.tile(np.reshape(np.mean(np.mean(XA,0),0)+10**-6,(1,1,-1)),(options['size'],options['size'],1))
    XAoffset = np.reshape(X_k_Aoffset,(options['size'],options['size']))
    
    # Update the weights
    weight = 1/(np.abs(XE_norm) + options['epsilon'])
    options['weight'] = weight * weight.size / np.sum(weight)
    
    # Calculate the mean flatfield and darkfield
    temp = np.mean(XA,2) - XAoffset
    flatfield = temp/np.mean(temp)
    darkfield = XAoffset
    return flatfield, darkfield, options

def _get_photobleach(D,flatfield,darkfield=None):
    """ Calculate the global effect of photobleaching for each image
    
    Using the original data, flatfield, and darkfield images, estimate the total contribution of photobleaching
    to an image in a series of images.

    Inputs:
        X_k_A - Flatfield approximation per pixel per image
        X_k_E - Error for every pixel
        X_k_Aoffset - Darkfield approximation per pixel per image
        options - optimization options
    Outputs:
        A_coeff - A 1xn matrix of photobleaching offsets, where n is the number of input images
    """
    # Initialize matrices
    D = np.reshape(D,(OPTIONS['size']*OPTIONS['size'],-1)).astype(np.float64)
    if darkfield is None:
        darkfield = np.zeros(flatfield.shape,dtype=np.float64)
    
    # Initialize weights and tolerances
    weights = np.ones(D.shape,dtype=np.float64)
    epsilon = np.float64(0.1)
    tol = np.float64(10**-6)
    
    # Run optimization exactly 5 times
    for r in range(5):
        # Calculate weights, offsets and coefficients
        W_idct_hat = np.reshape(flatfield,(-1,1))
        A_offset = np.reshape(darkfield,(-1,1))
        A_coeff = np.reshape(np.mean(D,0),(1,-1))
        
        # Initialization values and learning rates
        temp = np.linalg.svd(D,full_matrices=False,compute_uv=False)
        norm_two = np.float64(temp[0])
        mu = np.float64(12.5)/norm_two
        mu_bar = mu * 10**7
        rho = np.float64(1.5)
        ent1 = 1
        
        # Normalization factors
        d_norm = np.linalg.norm(D,'fro')
        
        # Initialize augmented representation and error
        A = np.zeros(D.shape,dtype=np.float64)
        E1 = np.zeros(D.shape,dtype=np.float64)
        Y1 = np.float64(0)
        
        # Run optimization
        iternum = 0
        converged = False
        while not converged:
            iternum += 1

            # Calculate augmented representation
            A = np.matmul(W_idct_hat,A_coeff) + A_offset
            
            # Calculate errors
            E1 = E1 + np.divide(D - A - E1 + np.multiply(1/mu,Y1),ent1)
            E1 = np.max(np.reshape(E1 - weights/(ent1*mu),(D.shape[0],D.shape[1],1)),-1,initial=0) + np.min(np.reshape(E1 + weights/(ent1*mu),(D.shape[0],D.shape[1],1)),-1,initial=0)
            
            # Calculate coefficients
            R1 = D-E1
            A_coeff = np.reshape(np.mean(R1,0),(1, -1)) - np.mean(A_offset)
            A_coeff[A_coeff<0] = 0      # pixel values are never negative
            
            # Loss
            Z1 = D - A - E1

            # Error updates
            Y1 = Y1 + mu*Z1
            
            # Update learning rate
            mu = np.min(mu*rho,initial=mu_bar)
            
            # Stop if below threshold
            stopCriterion = np.linalg.norm(Z1,'fro')/d_norm
            if stopCriterion < tol:
                converged = True
        
        # Update weights
        XE_norm = np.reshape(np.mean(A,0),(1,-1)) / E1
        weights = 1/np.abs(XE_norm + epsilon)
        weights = weights * weights.size/np.sum(weights)
        
    return A_coeff

if __name__ == "__main__":
    """ Initialize argument parser """
    parser = argparse.ArgumentParser(prog='basic', description='Calculate flatfield information for a single set of images.')

    """ Define the arguments """
    parser.add_argument('--inpDir',            # Name of the bucket
                        dest='inpDir',
                        type=str,
                        help='Path to input images.',
                        required=True)
    parser.add_argument('--darkfield',         # Path to the data within the bucket
                        dest='darkfield',
                        type=str,
                        help='If true, calculate darkfield contribution.',
                        required=False)
    parser.add_argument('--photobleach',       # Path to the data within the bucket
                        dest='photobleach',
                        type=str,
                        help='If true, calculates a photobleaching scalar.',
                        required=False)
    parser.add_argument('--inpRegex',          # Output directory
                        dest='inp_regex',
                        type=str,
                        help='Input file name pattern.',
                        required=False)
    parser.add_argument('--outDir',            # Output directory
                        dest='output_dir',
                        type=str,
                        help='The output directory for the flatfield images.',
                        required=True)
    parser.add_argument('--R',                 # Replicate
                        dest='R',
                        type=int,
                        help='The output directory for the flatfield images.',
                        required=False)
    parser.add_argument('--T',                 # Timepoint
                        dest='T',
                        type=int,
                        help='The output directory for the flatfield images.',
                        required=False)
    parser.add_argument('--C',                 # Channel
                        dest='C',
                        type=int,
                        help='The output directory for the flatfield images.',
                        required=False)
    
    """ Get the input arguments """
    args = parser.parse_args()

    fpath = args.inpDir
    get_darkfield = str(args.darkfield).lower() == 'true'
    output_dir = Path(args.output_dir).joinpath('images')
    metadata_dir = Path(args.output_dir).joinpath('metadata_files')
    inp_regex = args.inp_regex
    get_photobleach = str(args.photobleach).lower() == 'true'
    if args.R==None:
        R = 0
    else:
        R = args.R
    if args.T==None:
        T = 0
    else:
        T = args.T
    if args.C==None:
        C = 0
    else:
        C = args.C
    
    # Initialize the logger
    logging.basicConfig(format='%(asctime)s - %(name)s - Process [{0},{1},{2}] - %(levelname)s - %(message)s'.format(R,T,C),
                        datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger("BaSiC")
    logger.setLevel(logging.INFO)
    
    # Start the javabridge
    log_config = Path(__file__).parent.joinpath("log4j.properties")
    jutil.start_vm(args=["-Dlog4j.configuration=file:{}".format(str(log_config.absolute()))],class_path=bioformats.JARS)

    # Parse files into dictionary to acces by variable
    regex,variables = _parse_fpattern(inp_regex)
    files = _parse_files(fpath,regex,variables)

    # Load files and sort
    logger.info('Loading and sorting images...')
    img_stk,X,Y = _get_resized_image_stack(fpath,files[R][T][C]['file'])
    img_stk_sort = np.sort(img_stk)

    # Initialize options
    new_options = _initialize_options(img_stk_sort,get_darkfield,OPTIONS)

    # Initialize flatfield/darkfield matrices
    logger.info('Beginning flatfield estimation')
    flatfield_old = np.ones((new_options['size'],new_options['size']),dtype=np.float64)
    darkfield_old = np.random.normal(size=(new_options['size'],new_options['size'])).astype(np.float64)

    # Optimize and re-optimize until the change in values is below tolerance or a maximum number of iterations is reached
    for w in range(new_options['max_reweight_iterations']):
        # Optimize using inexact augmented Legrangian multiplier method using L1 loss
        A, E1, A_offset = _inexact_alm_l1(copy.deepcopy(img_stk_sort),new_options)
        
        # Calculate the flatfield/darkfield images and update training weights
        flatfield, darkfield, new_options = _get_flatfield_and_reweight(A,E1,A_offset,new_options)
        
        # Calculate the change in flatfield and darkfield images between iterations
        mad_flat = np.sum(np.abs(flatfield-flatfield_old))/np.sum(np.abs(flatfield_old))
        temp_diff = np.sum(np.abs(darkfield - darkfield_old))
        if temp_diff < 10**-7:
            mad_dark =0
        else:
            mad_dark = temp_diff/np.max(np.sum(np.abs(darkfield_old)),initial=10**-6)
        flatfield_old = flatfield
        darkfield_old = darkfield
        
        # Stop optimizing if the change in flatfield/darkfield is below threshold
        logger.info('Iteration {} loss: {}'.format(w+1,mad_flat))
        if np.max(mad_flat,initial=mad_dark) < new_options['reweight_tol']:
            break
    
    # Calculate photobleaching effects if specified
    if get_photobleach:
        pb = _get_photobleach(copy.deepcopy(img_stk),flatfield,darkfield)
    
    # Resize images back to original image size
    logger.info('Saving outputs...')
    flatfield = cv2.resize(flatfield,(Y,X),interpolation=cv2.INTER_CUBIC).astype(np.float32)
    if new_options['darkfield']:
        darkfield = cv2.resize(darkfield,(Y,X),interpolation=cv2.INTER_CUBIC).astype(np.float32)
        
    # Generate output image name based on filename pattern variables
    out_dict = {}
    if 'r' in variables:
        out_dict['r'] = R
    if 't' in variables:
        out_dict['t'] = T
    if 'c' in variables:
        out_dict['c'] = C
    base_output = _get_output_name(inp_regex,files[R][T][C],out_dict)
    
    # Export the flatfield image as a tiled tiff
    flatfield_out = base_output.replace('.ome.tif','_flatfield.ome.tif')
    bw = BioWriter(str(output_dir.joinpath(flatfield_out)))
    bw.pixel_type('float')
    bw.num_x(X)
    bw.num_y(Y)
    bw.write_image(np.reshape(flatfield,(Y,X,1,1,1)))
    bw.close_image()
    
    # Export the darkfield image as a tiled tiff
    if new_options['darkfield']:
        darkfield_out = base_output.replace('.ome.tif','_darkfield.ome.tif')
        bw = BioWriter(str(output_dir.joinpath(darkfield_out)))
        bw.pixel_type('float')
        bw.num_x(X)
        bw.num_y(Y)
        bw.write_image(np.reshape(darkfield,(Y,X,1,1,1)))
        bw.close_image()
        
    # Export the photobleaching components as csv
    if get_photobleach:
        offsets_out = base_output.replace('.ome.tif','_offsets.csv')
        with open(str(metadata_dir.joinpath(offsets_out)),'w') as fw:
            fw.write('file,offset\n')
            for f,o in zip(files[R][T][C]['file'],pb[0,:].tolist()):
                fw.write("{},{}\n".format(f,o))

    jutil.kill_vm()