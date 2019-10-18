import argparse, os, time, csv, logging, re, copy, cv2

from pathlib import Path
from bfio.bfio import BioReader,BioWriter
import numpy as np

VARIABLES = 'pxyzct'   # possible variables in input regular expression
STATICS = 'zt'         # dimensions usually processed separately
options = {'max_iterations': 500,
           'max_reweight_iterations': 10,
           'optimization_tol': 10**-6,
           'reweight_tol': 5*10**-4,
           'darkfield': False,
           'size': 128,
           'epsilon': 0.1
          }

# Initialize the logger    
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("main")
logger.setLevel(logging.INFO)

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
def _get_resized_image_stack(fpath,flist):
    # Load all the images in flist as a 3D stack of images
    
    #Initialize the output
    br = BioReader(str(Path(fpath).joinpath(flist[0])))
    X = br.num_x()
    Y = br.num_y()
    C = len(flist)
    img_stack = np.zeros((options['size'],options['size'],C),dtype=np.float64)
    
    # Load every image as a z-slice
    for ind,fname in zip(range(len(flist)),flist):
       logger.info("Loading ({}/{}): {}".format(ind+1,len(flist),fname))
       br = BioReader(str(Path(fpath).joinpath(fname)))
       I = np.squeeze(br.read_image())
       img_stack[:,:,ind] = cv2.resize(I,(options['size'],options['size']),interpolation=cv2.INTER_LINEAR).astype(np.float64)
       
    return img_stack,X,Y

def _dct2(D):
    # 2D DCT type-II
    # d = dct(D,norm='ortho',axis=0,type=2).astype(np.float64)
    # d = dct(d,norm='ortho',axis=1,type=2).astype(np.float64)
    d = cv2.dct(D).astype(np.float64)
    return d

def _idct2(D):
    # 2D DCT type-III - this is the inverse of the 2D DCT type-II
    # d = dct(D,norm='ortho',axis=0,type=3).astype(np.float64)
    # d = dct(d,norm='ortho',axis=1,type=3).astype(np.float64)
    d = cv2.dct(D,flags=cv2.DCT_INVERSE).astype(np.float64)
    return d

def _initialize_options(img_stack,get_darkfield,options):
    meanD = np.mean(img_stack,2)
    meanD = meanD/np.mean(meanD)
    weights = _dct2(meanD)
    new_options = copy.deepcopy(options)
    new_options['lambda'] = np.sum(np.abs(weights))/800
    new_options['lambda_darkfield'] = np.sum(np.abs(weights))/2000
    new_options['weight'] = np.ones(img_stack.shape,dtype=np.float64)
    new_options['darkfield_limit'] = 10**7
    new_options['darkfield'] = get_darkfield
    return new_options

def _inexact_alm_l1(D,options):
    # Transform data from image stack to image column vectors
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
    
    # Initialize gradient and weight normalization factors
    Y1 = np.float64(0)
    ent1 = np.float64(1)
    ent2 = np.float64(10)
    
    A1_hat = np.zeros(D.shape,dtype=np.float64)
    E1_hat = np.zeros(D.shape,dtype=np.float64)
    W_hat = _dct2(np.mean(np.reshape(A1_hat,(p,q,n)),2))
    mu = np.float64(12.5)/norm_two
    mu_bar = mu * 10**7
    rho = np.float64(1.5)
    d_norm = np.linalg.norm(D,'fro')
    A1_coeff = np.ones((1,n),dtype=np.float64)
    A_offset = np.zeros((m,1),dtype=np.float64)
    B1_uplimit = np.min(D)
    B1_offset = np.float64(0)
    
    # Main iteration loop
    iternum = 0
    converged = False
    while not converged:
        iternum += 1
        W_idct_hat = _idct2(W_hat)
        A1_hat = np.matmul(np.reshape(W_idct_hat,(m,1)),A1_coeff) + A_offset
        temp_W = np.divide(D - A1_hat - E1_hat + np.multiply(1/mu,Y1),ent1)
        
        temp_W = np.reshape(temp_W,(p,q,n))
        temp_W = np.mean(temp_W,2)
        W_hat = W_hat + _dct2(temp_W)
        W_hat = np.max(np.reshape(W_hat - options['lambda']/(ent1*mu),(p,q,1)),-1,initial=0) + np.min(np.reshape(W_hat + options['lambda']/(ent1*mu),(p,q,1)),-1,initial=0)
        W_idct_hat = _idct2(W_hat)
        
        A1_hat = np.matmul(np.reshape(W_idct_hat,(m,1)),A1_coeff) + A_offset
        
        E1_hat = E1_hat + np.divide(D - A1_hat - E1_hat + np.multiply(1/mu,Y1),ent1)
        E1_hat = np.max(np.reshape(E1_hat - options['weight']/(ent1*mu),(m,n,1)),-1,initial=0) + np.min(np.reshape(E1_hat + options['weight']/(ent1*mu),(m,n,1)),-1,initial=0)
        
        R1 = D-E1_hat
        A1_coeff = np.reshape(np.mean(R1,0)/np.mean(R1),(1,n))
        
        A1_coeff[A1_coeff<0] = 0
        
        if options['darkfield']:
            validA1coeff_idx = np.argwhere(A1_coeff<1)[:,1]
            R1_upper = R1[np.argwhere(np.reshape(W_idct_hat,(-1,1)).astype(np.float64)>(np.float64(np.mean(W_idct_hat))-np.float64(10**-5)))[:,0],:]
            R1_upper = np.mean(R1_upper[:,validA1coeff_idx],0)
            R1_lower = R1[np.argwhere(np.reshape(W_idct_hat,(-1,1))<np.mean(W_idct_hat)+np.float64(10**-5))[:,0],:]
            R1_lower = np.mean(R1_lower[:,validA1coeff_idx],0)
            B1_coeff = (R1_upper-R1_lower)/np.mean(R1)
            
            k = validA1coeff_idx.size
            
            temp1 = np.sum(np.square(A1_coeff[0,validA1coeff_idx]))
            temp2 = np.sum(A1_coeff[0,validA1coeff_idx])
            temp3 = np.sum(B1_coeff)
            temp4 = np.sum(A1_coeff[0,validA1coeff_idx]*B1_coeff)
            temp5 = temp2 * temp3 - k*temp4
            
            if temp5 == 0:
                B1_offset = np.float64(0)
            else:
                B1_offset = (temp1*temp3-temp2*temp4)/temp5
                
            B1_offset = np.max(B1_offset,initial=0)
            B1_offset = np.min(B1_offset,initial=B1_uplimit/(np.mean(W_idct_hat)+10**-7))
            B_offset = B1_offset * np.mean(W_idct_hat) - B1_offset*np.reshape(W_idct_hat,(-1,1))
            
            A1_offset = np.reshape(np.mean(R1[:,validA1coeff_idx],1),(-1,1)) - np.mean(A1_coeff[0,validA1coeff_idx]) * np.reshape(W_idct_hat,(-1,1))
            A1_offset = A1_offset - np.mean(A1_offset)
            A_offset = A1_offset - np.mean(A1_offset) - B_offset
            
            W_offset = _dct2(np.reshape(A_offset,(p,q)))
            W_offset = np.max(np.reshape(W_offset - options['lambda_darkfield']/(ent2*mu),(p,q,1)),-1,initial=0) \
                     + np.min(np.reshape(W_offset + options['lambda_darkfield']/(ent2*mu),(p,q,1)),-1,initial=0)
            A_offset = _idct2(W_offset)
            A_offset = np.reshape(A_offset,(-1,1))

            A_offset = np.max(np.reshape(A_offset - options['lambda_darkfield']/(ent2*mu),(A_offset.shape[0],A_offset.shape[1],1)),-1,initial=0) \
                     + np.min(np.reshape(A_offset + options['lambda_darkfield']/(ent2*mu),(A_offset.shape[0],A_offset.shape[1],1)),-1,initial=0)
            A_offset = A_offset + B_offset
            
        Z1 = D - A1_hat - E1_hat
        
        Y1 = Y1 + mu*Z1
        
        mu = np.min(mu*rho,initial=mu_bar)
        
        stopCriterion = np.linalg.norm(Z1,ord='fro')/d_norm
        
        logger.info('Iteration {}: stopCriterion {}'.format(iternum,stopCriterion))
        
        if stopCriterion < options['optimization_tol'] or iternum > options['max_iterations']:
            converged = True
        
    A_offset = A_offset + B1_offset * np.reshape(W_idct_hat,(-1,1))
    
    return A1_hat,E1_hat,A_offset

def _get_flatfield_and_reweight(X_k_A,X_k_E,X_k_Aoffset,options):
    # Reshape matrices back to original dimensions
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
    D = np.reshape(D,(options['size']*options['size'],-1)).astype(np.float64)
    if darkfield is None:
        darkfield = np.zeros(flatfield.shape,dtype=np.float64)
        
    weights = np.ones(D.shape,dtype=np.float64)
    epsilon = np.float64(0.1)
    tol = np.float64(10**-6)
    
    for r in range(5):
        W_idct_hat = np.reshape(flatfield,(-1,1))
        A_offset = np.reshape(darkfield,(-1,1))
        A1_coeff = np.reshape(np.mean(D,0),(1,-1))
            
        temp = np.linalg.svd(D,full_matrices=False,compute_uv=False)
        norm_two = np.float64(temp[0])
        mu = np.float64(12.5)/norm_two
        mu_bar = mu * 10**7
        rho = np.float64(1.5)
        
        d_norm = np.linalg.norm(D,'fro')
        ent1 = 1
        
        A1_hat = np.zeros(D.shape,dtype=np.float64)
        E1_hat = np.zeros(D.shape,dtype=np.float64)
        Y1 = np.float64(0)
        
        iternum = 0
        converged = False
        
        while not converged:
            iternum += 1
            A1_hat = np.matmul(W_idct_hat,A1_coeff) + A_offset
            
            E1_hat = E1_hat + np.divide(D - A1_hat - E1_hat + np.multiply(1/mu,Y1),ent1)
            E1_hat = np.max(np.reshape(E1_hat - weights/(ent1*mu),(D.shape[0],D.shape[1],1)),-1,initial=0) + np.min(np.reshape(E1_hat + weights/(ent1*mu),(D.shape[0],D.shape[1],1)),-1,initial=0)
            
            R1 = D-E1_hat
            
            A1_coeff = np.reshape(np.mean(R1,0),(1, -1)) - np.mean(A_offset)
            
            A1_coeff[A1_coeff<0] = 0
            
            Z1 = D - A1_hat - E1_hat
            
            Y1 = Y1 + mu*Z1
            
            mu = np.min(mu*rho,initial=mu_bar)
            
            stopCriterion = np.linalg.norm(Z1,'fro')/d_norm
            
            if stopCriterion < tol:
                converged = True
                
            logger.info('Iteration {}: stopCriterion {}'.format(iternum,stopCriterion))
        
        XE_norm = np.reshape(np.mean(A1_hat,0),(1,-1)) / E1_hat
        weights = 1/np.abs(XE_norm + epsilon)
        weights = weights * weights.size/np.sum(weights)
        
    return A1_coeff

def main():
    import bioformats
    import javabridge as jutil
    
    """ Initialize argument parser """
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='Calculate flatfield information from an image collection.')

    """ Define the arguments """
    parser.add_argument('--inpDir',               # Name of the bucket
                        dest='inpDir',
                        type=str,
                        help='Path to input images.',
                        required=True)
    parser.add_argument('--darkfield',                  # Path to the data within the bucket
                        dest='darkfield',
                        type=str,
                        help='If true, calculate darkfield contribution.',
                        required=False)
    parser.add_argument('--photobleach',                  # Path to the data within the bucket
                        dest='photobleach',
                        type=str,
                        help='If true, calculates a photobleaching scalar.',
                        required=True)
    parser.add_argument('--inpRegex',                 # Output directory
                        dest='inp_regex',
                        type=str,
                        help='Input file name pattern.',
                        required=False)
    parser.add_argument('--outDir',                 # Output directory
                        dest='output_dir',
                        type=str,
                        help='The output directory for the flatfield images.',
                        required=True)
    
    """ Get the input arguments """
    args = parser.parse_args()

    fpath = args.inpDir
    get_darkfield = args.darkfield == 'true'
    output_dir = args.output_dir
    inp_regex = args.inp_regex
    get_photobleach = args.photobleach == 'true'

    logger.info('input_dir = {}'.format(fpath))
    logger.info('get_darkfield = {}'.format(get_darkfield))
    logger.info('get_photobleach = {}'.format(get_photobleach))
    logger.info('inp_regex = {}'.format(inp_regex))
    logger.info('output_dir = {}'.format(output_dir))
    
    # Start the javabridge
    jutil.start_vm(class_path=bioformats.JARS)
    
    regex,variables = _parse_regex(inp_regex)
    files = _parse_files(fpath,regex,variables)
    
    zs = [key for key in files.keys()]
    zs.sort()
    for z in zs:
        ts = [t for t in files[z].keys()]
        ts.sort()
        for t in ts:
            cs = [c for c in files[z][t].keys()]
            cs.sort()
            for c in cs:
                logger.info('Loading images for [z][t][c]: {}'.format([z,t,c]))
                img_stk,X,Y = _get_resized_image_stack(fpath,files[z][t][c]['file'])
    
                logger.info('Sorting images...')
                img_stk_sort = np.sort(img_stk)
    
                new_options = _initialize_options(img_stk_sort,get_darkfield,options)

                logger.info('Estimating flatfield...')
                flatfield_old = np.ones((new_options['size'],new_options['size']),dtype=np.float64)
                darkfield_old = np.random.normal(size=(new_options['size'],new_options['size'])).astype(np.float64)
                for w in range(new_options['max_reweight_iterations']):
                    logger.info('Reweight iteration: {}'.format(w+1))
                    A1_hat, E1_hat, A_offset = _inexact_alm_l1(copy.deepcopy(img_stk_sort),new_options)
                    
                    logger.info('Calculating flatfield and reweighting...')
                    flatfield, darkfield, new_options = _get_flatfield_and_reweight(A1_hat,E1_hat,A_offset,new_options)
                    
                    logger.info('Calculating differences...')
                    mad_flat = np.sum(np.abs(flatfield-flatfield_old))/np.sum(np.abs(flatfield_old))
                    temp_diff = np.sum(np.abs(darkfield - darkfield_old))
                    if temp_diff < 10**-7:
                        mad_dark =0
                    else:
                        mad_dark = temp_diff/np.max(np.sum(np.abs(darkfield_old)),initial=10**-6)
                    flatfield_old = flatfield
                    darkfield_old = darkfield
                    
                    if np.max(mad_flat,initial=mad_dark) < new_options['reweight_tol']:
                        break
                    
                if get_photobleach:
                    logger.info('Calculating photobleach offsets...')
                    pb = _get_photobleach(copy.deepcopy(img_stk),flatfield,darkfield)
                
                logger.info('Resizing images...')
                flatfield = cv2.resize(flatfield,(Y,X),interpolation=cv2.INTER_CUBIC).astype(np.float32)
                if options['darkfield']:
                    darkfield = cv2.resize(darkfield,(Y,X),interpolation=cv2.INTER_CUBIC).astype(np.float32)
                    
                out_dict = {}
                if z in variables:
                    out_dict['z'] = z
                if t in variables:
                    out_dict['t'] = t
                if c in variables:
                    out_dict['c'] = c
                base_output = _get_output_name(inp_regex,out_dict)
                
                logger.info('Saving flatfield image...')
                logger.info('Base output name: {}'.format(base_output))
                flatfield_out = base_output.replace('.ome.tif','_flatfield.ome.tif')
                logger.info('flatfield output name: {}'.format(flatfield_out))
                logger.info('Fullpath: {}'.format(str(Path(output_dir).joinpath(flatfield_out))))
                bw = BioWriter(str(Path(output_dir).joinpath(flatfield_out)))
                bw.pixel_type('float')
                bw.num_x(X)
                bw.num_y(Y)
                bw.write_image(flatfield)
                
                if get_darkfield:
                    logger.info('Saving darkfield image...')
                    darkfield_out = base_output.replace('.ome.tif','_darkfield.ome.tif')
                    bw = BioWriter(str(Path(output_dir).joinpath(darkfield_out)))
                    bw.pixel_type('float')
                    bw.num_x(X)
                    bw.num_y(Y)
                    bw.write_image(darkfield)
                    
                if get_photobleach:
                    logger.info('Saving photobleach offsets...')
                    
    jutil.kill_vm()
        
if __name__ == "__main__":
    import bioformats
    import javabridge as jutil
    from matplotlib import pyplot as plt
    
    # Start the javabridge
    jutil.start_vm(class_path=bioformats.JARS)
    
    fpath = "/media/nick/My2TBSSD/BrainData/images"
    inp_regex = "S1_R1_C1-C11_A1_y{yyy}_x{xxx}_c000.ome.tif"
    
    # Start the javabridge
    jutil.start_vm(class_path=bioformats.JARS)
    
    regex,variables = _parse_regex(inp_regex)
    files = _parse_files(fpath,regex,variables)
    
    get_darkfield = False
    get_photobleach = False
    output_dir = '.'
    
    zs = [key for key in files.keys()]
    zs.sort()
    for z in zs:
        ts = [t for t in files[z].keys()]
        ts.sort()
        for t in ts:
            cs = [c for c in files[z][t].keys()]
            cs.sort()
            for c in cs:
                logger.info('Loading images for [z][t][c]: {}'.format([z,t,c]))
                img_stk,X,Y = _get_resized_image_stack(fpath,files[z][t][c]['file'])
    
                logger.info('Sorting images...')
                img_stk_sort = np.sort(img_stk)
    
                new_options = _initialize_options(img_stk_sort,get_darkfield,options)

                logger.info('Estimating flatfield...')
                flatfield_old = np.ones((new_options['size'],new_options['size']),dtype=np.float64)
                darkfield_old = np.random.normal(size=(new_options['size'],new_options['size'])).astype(np.float64)
                for w in range(new_options['max_reweight_iterations']):
                    logger.info('Reweight iteration: {}'.format(w+1))
                    A1_hat, E1_hat, A_offset = _inexact_alm_l1(copy.deepcopy(img_stk_sort),new_options)
                    
                    logger.info('Calculating flatfield and reweighting...')
                    flatfield, darkfield, new_options = _get_flatfield_and_reweight(A1_hat,E1_hat,A_offset,new_options)
                    
                    logger.info('Calculating differences...')
                    mad_flat = np.sum(np.abs(flatfield-flatfield_old))/np.sum(np.abs(flatfield_old))
                    temp_diff = np.sum(np.abs(darkfield - darkfield_old))
                    if temp_diff < 10**-7:
                        mad_dark =0
                    else:
                        mad_dark = temp_diff/np.max(np.sum(np.abs(darkfield_old)),initial=10**-6)
                    flatfield_old = flatfield
                    darkfield_old = darkfield
                    
                    if np.max(mad_flat,initial=mad_dark) < new_options['reweight_tol']:
                        break
                    
                if get_photobleach:
                    logger.info('Calculating photobleach offsets...')
                    pb = _get_photobleach(copy.deepcopy(img_stk),flatfield,darkfield)
                
                logger.info('Resizing images...')
                flatfield = cv2.resize(flatfield,(Y,X),interpolation=cv2.INTER_CUBIC).astype(np.float32)
                if options['darkfield']:
                    darkfield = cv2.resize(darkfield,(Y,X),interpolation=cv2.INTER_CUBIC).astype(np.float32)
                    
                out_dict = {}
                if z in variables:
                    out_dict['z'] = z
                if t in variables:
                    out_dict['t'] = t
                if c in variables:
                    out_dict['c'] = c
                base_output = _get_output_name(inp_regex,out_dict)
                
                logger.info('Saving flatfield image...')
                logger.info('Base output name: {}'.format(base_output))
                flatfield_out = base_output.replace('.ome.tif','_flatfield.ome.tif')
                logger.info('flatfield output name: {}'.format(flatfield_out))
                logger.info('Fullpath: {}'.format(str(Path(output_dir).joinpath(flatfield_out))))
                bw = BioWriter(str(Path(output_dir).joinpath(flatfield_out)))
                bw.pixel_type('float')
                bw.num_x(X)
                bw.num_y(Y)
                bw.write_image(flatfield)
                
                if get_darkfield:
                    logger.info('Saving darkfield image...')
                    darkfield_out = base_output.replace('.ome.tif','_darkfield.ome.tif')
                    bw = BioWriter(str(Path(output_dir).joinpath(darkfield_out)))
                    bw.pixel_type('float')
                    bw.num_x(X)
                    bw.num_y(Y)
                    bw.write_image(darkfield)
                    
                if get_photobleach:
                    logger.info('Saving photobleach offsets...')
    
    #main()