from skimage import measure
from skimage.measure import shannon_entropy
from skimage.segmentation import clear_border
from scipy.stats import skew
from scipy.stats import kurtosis as kurto
from scipy.stats import mode as modevalue
from scipy.spatial import ConvexHull
from bfio import BioReader
from scipy import special,ndimage,signal
import numpy as np
import pandas as pd
import skimage.morphology
import argparse
import logging
import os
import math
import filepattern
import itertools
import tempfile

# Initialize the logger
logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
					datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("main")
logger.setLevel(logging.INFO)

def read(img_file):
    """Read the .ome.tif image using BioReader.
    
    Args:
        img_directory (str): Path to the directory containing the input images.
        
    Returns:
        Array of the image and the embedded unit in the metadata if present else it will be none.
        
    """
    image_bfio = BioReader(img_file)
    #Get embedded units from metadata (physical size)
    img_unit = image_bfio.ps_y[1]
    #Image shape
    bfshape = image_bfio.shape
    datatype = np.dtype(image_bfio.dtype)
    #Define chunksize
    chunk_size = [256,256,256]
    xsplits = list(np.arange(0, bfshape[0], chunk_size[0]))
    xsplits.append(bfshape[0])
    ysplits = list(np.arange(0, bfshape[1], chunk_size[1]))
    ysplits.append(bfshape[1])
    zsplits = list(np.arange(0, bfshape[2], chunk_size[2]))
    zsplits.append(bfshape[2])
    all_identities = []
    xb = np.array([])
    with tempfile.TemporaryDirectory() as temp_dir:
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir, exist_ok=True)
        for y in range(len(ysplits)-1):
            for x in range(len(xsplits)-1):
                for z in range(len(zsplits)-1):
                    start_y, end_y = (ysplits[y], ysplits[y+1])
                    start_x, end_x = (xsplits[x], xsplits[x+1])
                    start_z, end_z = (zsplits[z], zsplits[z+1])
                    volume = image_bfio[start_x:end_x,start_y:end_y,start_z:end_z]
                    volume = volume.flatten()
                    xb=np.append(xb,volume)
    #Reshape array based on image shape
    img_data = np.reshape(xb,[bfshape[0],bfshape[1],bfshape[2]])
    label_image = img_data.astype(int)
    logger.info('Done reading the file: {}'.format(img_file.name))
    return label_image, img_unit

def strel_disk(radius):
    """Create a disk structuring element for morphological operations.
    
    Args:
        radius(int) - Radius of the disk.
        
    Returns:
        Array containing morphological dilation of an image.
    """
    iradius = int(radius)
    x, y = np.mgrid[-iradius : iradius + 1, -iradius : iradius + 1]
    radius2 = radius * radius
    strel = np.zeros(x.shape)
    strel[x * x + y * y <= radius2] = 1
    return strel

def neighbors_find(lbl_img, pixeldistance):
    """Calculate the number of objects within d pixels of object n.

    Args:
        lbl_image (ndarray): Labeled image array.
        boxsize (int): Box size value.
        pixeldistance (int): Pixel distance value.

    Returns:
        An array showing the number of neighbors touching the object for each object in labeled image. 

    Note:
        Number_of_Neighbors = neighbors_find(label_image, boxsize, pixeldistance=None)
        Computes the number of objects within 5 pixels of each object.

    """
    neighbors=[]
    dimensions = len(lbl_img.shape)
    nobjects = np.max(lbl_img)
    neighbor_count = np.zeros((nobjects,))
    if nobjects > 1:
        object_indexes = np.arange(nobjects, dtype=np.int32) + 1
        # Make the structuring element for dilation
        if dimensions == 2:
            strel = strel_disk(pixeldistance)
        else:
            strel = skimage.morphology.ball(pixeldistance)
        
        if dimensions == 2:
            i, j = np.mgrid[0 : lbl_img.shape[0], 0 : lbl_img.shape[1]]

            minimums_i, maximums_i, _, _ = ndimage.extrema(i, lbl_img, object_indexes)
            minimums_j, maximums_j, _, _ = ndimage.extrema(j, lbl_img, object_indexes)

            minimums_i = np.maximum(minimums_i - pixeldistance, 0).astype(int)
            maximums_i = np.minimum(maximums_i + pixeldistance + 1, lbl_img.shape[0] ).astype(int)
            minimums_j = np.maximum(minimums_j - pixeldistance, 0).astype(int)
            maximums_j = np.minimum(maximums_j + pixeldistance + 1, lbl_img.shape[1] ).astype(int)
        else:
            k, i, j = np.mgrid[0 : lbl_img.shape[0], 0 : lbl_img.shape[1], 0 : lbl_img.shape[2]]

            minimums_k, maximums_k, _, _ = ndimage.extrema(k, lbl_img, object_indexes)
            minimums_i, maximums_i, _, _ = ndimage.extrema(i, lbl_img, object_indexes)
            minimums_j, maximums_j, _, _ = ndimage.extrema(j, lbl_img, object_indexes)

            minimums_k = np.maximum(minimums_k - pixeldistance, 0).astype(int)
            maximums_k = np.minimum(maximums_k + pixeldistance + 1, lbl_img.shape[0]).astype(int)
            minimums_i = np.maximum(minimums_i - pixeldistance, 0).astype(int)
            maximums_i = np.minimum(maximums_i + pixeldistance + 1, lbl_img.shape[1]).astype(int)
            minimums_j = np.maximum(minimums_j - pixeldistance, 0).astype(int)
            maximums_j = np.minimum(maximums_j + pixeldistance + 1, lbl_img.shape[2]).astype(int)

        # Loop over all objects
        for object_number in range(0,len(maximums_j)):
            index = object_number - 1
            npatch = lbl_img[
                minimums_k[index] : maximums_k[index],
                minimums_i[index] : maximums_i[index],
                minimums_j[index] : maximums_j[index],
            ]

            # Find the neighbors
            patch_mask = npatch == (index + 1)
            if pixeldistance <= 5:
                extended = ndimage.binary_dilation(patch_mask, strel)
            else:
                extended = (
                    signal.fftconvolve(patch_mask, strel, mode="same") > 0.5
                )
            neighbors = np.unique(npatch[extended])
            neighbors = neighbors[neighbors != 0]
            neighbors = neighbors[neighbors != object_number]
            nc = len(neighbors)+1
            neighbor_count[index] = nc
        return neighbor_count

def convexhull(lbl_image):
    """ Calculates convex hull of 3D image.
    
    Args:
        lbl_image(ndarray): 3D labeled image.
        
    Returns:
        convex(scipy.spatial.ConvexHull): 3D convex hull of object.
    """
    bw = lbl_image > 0
    (x, y, z) = np.where(bw)
    x = np.concatenate([x]*2)
    y = np.concatenate([y]*2)
    z = np.concatenate([z]*2)
    # To transform the indexes from above into real coordinates we need to make the region + 0.5 greater in every direction
    x_all = []
    y_all = []
    z_all = []
    shifts = np.array([[0.5,0.5],[-0.5,0.5],[0.5,-0.5],[-0.5,-0.5]])    
    for shift in shifts:
        x_shifted = x + shift[0]
        x_all.append(x_shifted)
        y_shifted = y + shift[1]
        y_all.append(y_shifted)
        z_all.append(z)
    x_all = np.concatenate(x_all)
    y_all = np.concatenate(y_all)
    z_all = np.concatenate(z_all)
    points_3d = np.column_stack([x_all,y_all,z_all])
    convex = ConvexHull(points_3d)
    return convex

def rotate(pts, R):
    """ Applies rotation matrix ``R`` to pointcloud ``pts``.
    
    Args:
        pts (np.ndarray): Points to be rotated of shape ``(n,d)``, where ``n`` is the number of points and ``d`` the number of dimensions.
        R (np.ndarray): Rotation matrix of shape ``(d,d)``.
        
    Returns:
        pts_rot (np.ndarray): Rotated points.
    
    """   
    pts_rot = np.dot(pts, R)
    return pts_rot

def Rx(angle_in_degrees):
    """ Returns 3D rotation matrix for rotating around x-axis.
    
    Args:
        angle_in_degrees (float): Rotation angle around x-axis in degrees.
        
    Returns:
        Rx (np.ndarray): 3D Rotation matrix.
    
    """
    angle_in_rad = angle_in_degrees/180 * np.pi
    Rx = np.array([[1,0,0],[0,np.cos(angle_in_rad),-np.sin(angle_in_rad)],[0,np.sin(angle_in_rad),np.cos(angle_in_rad)]])
    return Rx

def project_to_xz(p_3d):
    """ Projects 3D points to xy plane.
    
    Args:
        p_3d (np.ndarray): 3D pointcloud in shape ``(n,3)``, where ``n`` is the number of points.
        
    Returns:
        p_2d (np.ndarray): Projected pointcloud in shape ``(n,2)``, where ``n`` is the number of points.
    
    """
    p_2d = p_3d[:,[0,2]]
    return p_2d

def R_2d(angle_in_degrees):
    """ Returns 2D rotation matrix.
    
    Args:
        angle_in_degrees (float): Rotation angle in degrees.
        
    Returns:
        R (np.ndarray): 2D Rotation matrix.

    """
    angle_in_rad = angle_in_degrees/180 * np.pi
    R = np.array([[np.cos(angle_in_rad), -np.sin(angle_in_rad)],[np.sin(angle_in_rad), np.cos(angle_in_rad)]])
    return R

def caliper(pts_2d, axis=1):
    """ Calculates 2d caliper from pointcloud.
    
    Args:
        pts_2d (np.ndarry, shape=(n,2)): 2D pointcloud.
        axis (float, 0 or 1, optional): axis.
        
    Returns:
        caliper (float): Caliper distance.
    
    """
    caliper = np.max(pts_2d[:,axis]) - np.min(pts_2d[:,axis])
    return caliper

def feret_diameter(convex):
    """ Calculates 3D feret diameter (min, max).
    
    Args:
        convex(scipy.spatial.ConvexHull): 3D convex hull of object.
        
    Returns:
        max_feret_3d(float): Max feret diameter.
        min_feret_3d(float): Min feret diameter.
    
    """
    p_3d = convex.points[convex.vertices]

    # feret diameters
    angles_x = np.arange(1,181)
    angles_y = np.arange(1,181)

    n_angles_x = len(angles_x)
    n_angles_y = len(angles_y)

    calipers = np.zeros((n_angles_x, n_angles_y))

    for i, a_x in enumerate(angles_x):
        p_3d_rot = rotate(p_3d, Rx(a_x))
        p_2d = project_to_xz(p_3d_rot)
        for j, a_y in enumerate(angles_y):
            p_2d_rot = rotate(p_2d, R_2d(a_y))
            calipers[i,j] = caliper(p_2d_rot, axis=1)
    feret_diameter = calipers.ravel()
    return feret_diameter

#Sort according to Distance/Radius (Min To Max)
def qsort(distance,x_grid,y_grid,z_grid):
    startpoint = 0
    endpoint = len(distance)
    output = np.concatenate((distance,x_grid,y_grid,z_grid),axis=1).T
    if(startpoint < endpoint):
        flag = startpoint
        for j in range((startpoint+1),(endpoint)):
            if(output[0,startpoint]>output[0,j]):
                flag = flag+1
                temp = output[:,flag].copy()
                output[:,flag] = output[:,j]
                output[:,j] = temp
        temp = output[:,startpoint].copy()
        output[:,startpoint] = output[:,flag]
        output[:,flag] = temp
        output_0=np.matrix(output[:,startpoint:flag])
        output[:,startpoint:flag] = qsort(output_0[0,:].T,output_0[1,:].T,output_0[2,:].T,output_0[3,:].T)
        output_flag0=np.matrix(output[:,flag+1:endpoint])
        output[:,flag+1:endpoint] = qsort(output_flag0[0,:].T,output_flag0[0,:].T,output_flag0[0,:].T,output_flag0[0,:].T)
    return output
    
#Calculate Legendre
def legendre(n,X) :
    res = []
    for m in range(n+1):
        res.append(special.lpmv(m,n,X))
    return res
    
#Compute spherical harmonics
def spharm(L,M,THETA,PHI):
    if ((L==0) and (M ==0) and (THETA==0) and (PHI==0)):
        L=2
        M=1
        THETA = math.pi/4
        PHI = math.pi/4
        
    if L<M:
        raise ValueError('The ORDER (M) must be less than or eqaul to the DEGREE(L).')
    
    Lmn=np.array(legendre(L,np.cos(PHI)))
    
    if L!=0:
        Lmn=np.squeeze(Lmn[M,...])

    a1=((2*L+1)/(4*math.pi))
    a2=math.factorial(L-M)/math.factorial(L+M)
    C=np.sqrt(a1*a2)
    Ymn=C*Lmn*math.e**(M*THETA*1j)
    return Ymn
    
#Get Spherical descriptors
def spherical_descriptors(file):
    angle = math.pi/6
    verti_trans, faces_ellip,_,_ = measure.marching_cubes(file,0)
    verti_x = np.matrix(verti_trans[0].ravel())
    verti_y = np.matrix(verti_trans[1].ravel())
    verti_z = np.matrix(verti_trans[2].ravel())
    
    # rotated
    c, s = np.cos(angle), np.sin(angle)
    R = np.matrix([[c, -s], [s, c]])
    result = R* (np.array([verti_x,verti_y]))
    verti_x = result[0,:].T
    verti_y = result[1,:].T
    verti_z = verti_z.T
    
    #align coordinates
    verti_x = verti_x-np.min(verti_x)
    verti_y = verti_y-np.min(verti_y)
    verti_z = verti_z-np.min(verti_z)

    #normalize to 1 and rasterize to 2Rx2Rx2R voxel grid
    max_value = np.max([np.max(verti_x),np.max(verti_y),np.max(verti_z)])
    R = 32
    x1 = np.round(verti_x/max_value*(2*R-1))
    y1 = np.round(verti_y/max_value*(2*R-1))
    z1 = np.round(verti_z/max_value*(2*R-1))
    x_grid = []
    y_grid = []
    z_grid = []
    grid = np.zeros((2*R,2*R,2*R))
    n_points = len(x1)
    for j in range(n_points):
        if(grid[int(x1[j]),int(y1[j]),int(z1[j])]==0):
            #register
            grid[int(x1[j]),int(y1[j]),int(z1[j])]=1
            x_grid.append(x1[j])
            y_grid.append(y1[j])
            z_grid.append(z1[j])

    #get center of mass
    x_center =np.mean(x_grid)
    y_center =np.mean(y_grid)
    z_center =np.mean(z_grid)
    x_grid = np.array(x_grid).ravel() - x_center
    y_grid = np.array(y_grid).ravel() - y_center
    z_grid = np.array(z_grid).ravel() - z_center
    
    #scale and make the average distance to center of mass is R/2
    dist = np.sqrt((x_grid)**2 + (y_grid)**2 + (z_grid)**2)
    mean_dist = np.mean(dist)
    scale_ratio = (R/2)/mean_dist
    x_grid_sr = x_grid * scale_ratio
    y_grid_sr = y_grid * scale_ratio
    z_grid_sr = z_grid * scale_ratio
    final_dist = np.sqrt((x_grid_sr)**2 + (y_grid_sr)**2 + (z_grid_sr)**2)
    distance=np.matrix(final_dist).T
    x_grid_val=np.matrix(x_grid_sr).T
    y_grid_val=np.matrix(y_grid_sr).T
    z_grid_val=np.matrix(z_grid_sr).T

    # qsort function
    output = qsort(distance,x_grid_val,y_grid_val,z_grid_val)
    output1 = np.array(output).T
    dist_vector = output1[:,0]
    
    #Get phi value
    phi   = np.arctan2(output1[:,2],output1[:,1])
    
    #Get theta value
    aa= output1[:,3]/dist_vector
    theta = np.arccos(aa)
    max_l = 16
    max_r = 32
    sph = np.zeros((max_r,max_l))
    
    #Get shape descriptors
    for idx_n in range(0,len(dist_vector)+1,100):
        idx_r = math.ceil(dist_vector[idx_n])
        for idx_l in range(0,(max_l)):
            Y_ml = 0
            for idx_m in range(-idx_l,idx_l+1):
                if(idx_m>=0):
                    Y_ml = Y_ml+spharm(idx_l,idx_m,theta[idx_n],phi[idx_n])
                else:
                    Y_temp = spharm(idx_l,-idx_m,theta[idx_n],phi[idx_n])
                    Y_ml = Y_ml+(-1)**(-idx_m) * np.conj(Y_temp)
            F_lr = Y_ml
            sph[idx_r-1,idx_l]=sph[idx_r-1,idx_l] + abs(F_lr) ** 2
    sph_des=np.sqrt(sph)
    return sph_des

def feature_extraction(features,
                        embeddedpixelsize,
                        unitLength,
                        pixelsPerunit,
                        pixelDistance,
                        channel,
                        intensity_image=None,
                        img_emb_unit=None,
                        label_image=None,
                        seg_file_names1=None,
                        int_file_name=None):
    """Calculate shape and intensity based features.

    Args:
        label_image (ndarray): Labeled image array.
        features (string): List of features to be extracted.
        seg_file_names1 (string): Filename of the labeled image.
        intfile_name (string): Filename of the intensity image.
        embeddedpixelsize (boolean): If true will consider units from metadata.
        img_emb_unit (string): Units embedded in the metadata.
        unitLength (string): Required units for the extracted features.
        pixelsPerunit (int): Pixels per unit for the metric mentioned in unitLength.
        intensity_image (ndarray): Intensity image array.
        pixelDistance (int): Distance between pixels to calculate the neighbors touching the object and default valus is 5.
        channel (int): Channel of the image.

    Returns:
        Dataframe containing the features extracted and the filename of the labeled image.

    """ 
    df_insert = pd.DataFrame([])
    if pixelDistance is None:
        pixelDistance = 5

    def volume(seg_img, units, *args):
        """Calculate volume for all the regions of interest in the image."""          
        data_dict1 = [region.area for region in regions]
        if unitLength and not embeddedpixelsize:
            data_dict = [dt_pixel / pixelsPerunit**3 for dt_pixel in data_dict1]
        else:
            data_dict = data_dict1
        logger.debug('Completed extracting volume for ' + seg_file_names1.name)
        return data_dict

    def convex_volume(seg_img, units, *args):
        """Calculate convex_volume for all the regions of interest in the image."""          
        data_dict1 = [region.convex_area for region in regions]
        if unitLength and not embeddedpixelsize:
            data_dict = [dt_pixel / pixelsPerunit**3 for dt_pixel in data_dict1]
        else:
            data_dict = data_dict1
        logger.debug('Completed extracting convex volume for ' + seg_file_names1.name)
        return data_dict

    def centroid_x(*args):
        """Calculate centroidx for all the regions of interest in the image."""
        centroid_value = [str(region.centroid) for region in regions]
        cent_x= [cent.split(',') for cent in centroid_value]
        data_dict = [centroid_x[1].replace(')','') for centroid_x in cent_x]
        logger.debug('Completed extracting centroid_column for ' + seg_file_names1.name)
        return data_dict

    def centroid_y(*args):
        """Calculate centroidy for all the regions of interest in the image."""
        centroid_value = [str(region.centroid) for region in regions]
        cent_y = [cent.split(',') for cent in centroid_value]
        data_dict = [centroid_y[0].replace('(','') for centroid_y in cent_y]
        logger.debug('Completed extracting centroid_column for ' + seg_file_names1.name)
        return data_dict

    def equivalent_diameter(seg_img, units, *args):
        """Calculate equivalent_diameter for all the regions of interest in the image."""
        data_dict1 = [region.equivalent_diameter for region in regions]
        if unitLength and not embeddedpixelsize:
            data_dict = [dt_pixel / pixelsPerunit for dt_pixel in data_dict1]
        else:
            data_dict = data_dict1
        logger.debug('Completed extracting equivalent diameter for ' + seg_file_names1.name)
        return data_dict

    def euler_number(*args):
        """Calculate euler_number for all the regions of interest in the image."""
        data_dict = [region.euler_number for region in regions]
        logger.debug('Completed extracting euler number for ' + seg_file_names1.name)
        return data_dict

    def major_axis_length(seg_img, units, *args):
        """Calculate major_axis_length for all the regions of interest in the image."""
        data_dict1 = [region.major_axis_length for region in regions]
        if unitLength and not embeddedpixelsize:
            data_dict = [dt_pixel / pixelsPerunit for dt_pixel in data_dict1]
        else:
            data_dict = data_dict1
        logger.debug('Completed extracting major axis length for ' + seg_file_names1.name)
        return data_dict

    def minor_axis_length(seg_img, units, *args):
        """Calculate minor_axis_length for all the regions of interest in the image."""
        data_dict1 = [region.minor_axis_length for region in regions]
        if unitLength and not embeddedpixelsize:
            data_dict = [dt_pixel / pixelsPerunit for dt_pixel in data_dict1]
        else:
            data_dict = data_dict1
        logger.debug('Completed extracting minor axis length for ' + seg_file_names1.name)
        return data_dict
    
    def gyration_tensor(seg_img):
        """Calculate gyration_tensor for all the regions of interest in the image."""
        imgs = [region.image for region in regions]
        data_dict=[]
        for seg in imgs:
            x,y,z = np.where(seg>0)
            data = np.c_[x,y,z]
            center = np.mean(data, 0)
            coord = data - center
            gyr_tensor = np.dot(coord.transpose(), coord)/float(len(coord))
            data_dict.append(gyr_tensor)
        return data_dict
    
    def asphericity(seg_img,*args):
        """Calculate asphericity for all the regions of interest in the image."""
        eigen = np.linalg.eigvalsh(gyration_tensor(seg_img)) 
        data_dict =[e[2] - 0.5*(e[0] + e[1]) for e in eigen]
        logger.debug('Completed extracting asphericity for ' + seg_file_names1.name)
        return data_dict
    
    def acylindricity(seg_img,*args):
        """Calculate acylindricity for all the regions of interest in the image."""
        eigen = np.linalg.eigvalsh(gyration_tensor(seg_img))
        data_dict = [e[1]-e[0] for e in eigen]
        logger.debug('Completed extracting acylindricity for ' + seg_file_names1.name)
        return data_dict
    
    def anisotrophy(seg_img,*args):
        """Calculate shape_anisotrophy for all the regions of interest in the image."""
        eigen = np.linalg.eigvalsh(gyration_tensor(seg_img))
        data_dict = [(((e[2] - 0.5*(e[0] + e[1]))**2) + (3/4*((e[1]-e[0])**2)))/((e[0]*e[0])+(e[1]*e[1])+(e[2]*e[2])) for e in eigen]
        logger.debug('Completed extracting anisotrophy for ' + seg_file_names1.name)
        return data_dict
        
    def orientation(seg_img,*args):
        """Calculate orientation for all the regions of interest in the image."""
        data_dict=[]
        eigen = np.linalg.eigvalsh(gyration_tensor(seg_img))
        for e in eigen:
            x = e[0]
            y = e[1]
            z = e[2]
            if x - z == 0:
                if y < 0:
                    orient = -math.pi / 4
                else:
                    orient =  math.pi / 4
            else:
                orient =  0.5 * math.atan2(-2 * y, z - x)
            data_dict.append(orient)
        logger.debug('Completed extracting orientation for ' + seg_file_names1.name)
        return data_dict
    
    def minimum_principal_moment(seg_img,*args):
        """Calculate minimum principal moment for all the regions of interest in the image."""
        eigen= np.linalg.eigvalsh(gyration_tensor(seg_img))
        data_dict = [np.min(e) for e in eigen]
        logger.debug('Completed extracting minimum principal moment for ' + seg_file_names1.name)
        return data_dict
    
    def maxferet(seg_img, units, *args):
       """Calculate maxferet for all the regions of interest in the image."""
       convex_image = [convexhull(region.image) for region in regions]
       feret = [feret_diameter(convex) for convex in convex_image]
       data_dict1= [np.max(f) for f in feret]
       if unitLength and not embeddedpixelsize:
           data_dict = [dt_pixel / pixelsPerunit for dt_pixel in data_dict1]
       else:
           data_dict = data_dict1
       logger.debug('Completed extracting maxferet for ' + seg_file_names1.name)
       return data_dict
   
    def solidity(seg_img,*args):
        """Calculate solidity for all the regions of interest in the image."""
        data_dict = [region.solidity for region in regions]
        logger.debug('Completed extracting solidity for ' + seg_file_names1.name)
        return data_dict

    def mean_intensity(*args):
        """Calculate mean_intensity for all the regions of interest in the image."""
        if label_image is not None:
            intensity_images = [region.intensity_image for region in regions]
            imgs = [region.image for region in regions]
            data_dict = [(np.mean(intensity[seg])) for intensity, seg in zip(intensity_images, imgs)]
        else:
            data_dict =np.mean(intensity_image.reshape(-1))
        logger.debug('Completed extracting mean intensity for ' + int_file_name)
        return data_dict

    def max_intensity(*args):
        """Calculate maximum intensity for all the regions of interest in the image."""
        if label_image is not None:
            intensity_images = [region.intensity_image for region in regions]
            imgs = [region.image for region in regions]
            data_dict = [int((np.max(intensity[seg]))) for intensity, seg in zip(intensity_images, imgs)]
        else:
            data_dict = np.max(intensity_image.reshape(-1))
        logger.debug('Completed extracting maximum intensity for ' + int_file_name)
        return data_dict

    def min_intensity(*args):
        """Calculate minimum intensity for all the regions of interest in the image."""
        if label_image is not None:
            intensity_images = [region.intensity_image for region in regions]
            imgs = [region.image for region in regions]
            data_dict = [int((np.min(intensity[seg]))) for intensity, seg in zip(intensity_images, imgs)]
        else:
            data_dict = np.min(intensity_image.reshape(-1))
        logger.debug('Completed extracting minimum intensity for ' + int_file_name)
        return data_dict

    def median(*args):
        """Calculate median for all the regions of interest in the image."""
        if label_image is not None:
            intensity_images = [region.intensity_image for region in regions]
            imgs = [region.image for region in regions]
            data_dict = [int((np.median(intensity[seg]))) for intensity, seg in zip(intensity_images, imgs)]
        else:
            data_dict = np.median(intensity_image.reshape(-1))
        logger.debug('Completed extracting median for ' + int_file_name)
        return data_dict

    def mode(*args):
        """Calculate mode for all the regions of interest in the image."""
        if label_image is not None:
            intensity_images = [region.intensity_image for region in regions]
            imgs = [region.image for region in regions]
            mode_list = [modevalue(intensity[seg])[0] for intensity, seg in zip(intensity_images, imgs)]
            data_dict = [str(mode_ls)[1:-1] for mode_ls in mode_list]
        else:
            data_dict = modevalue(intensity_image.reshape(-1))[0]
        logger.debug('Completed extracting mode for ' + int_file_name)
        return data_dict

    def standard_deviation(*args):
        """Calculate standard deviation for all the regions of interest in the image."""
        if label_image is not None:
            intensity_images = [region.intensity_image for region in regions]
            imgs = [region.image for region in regions]
            data_dict = [(np.std(intensity[seg])) for intensity, seg in zip(intensity_images, imgs)]
        else:
            data_dict= np.std(intensity_image.reshape(-1))
        logger.debug('Completed extracting standard deviation for ' + int_file_name)
        return data_dict

    def skewness(*args):
        """Calculate skewness for all the regions of interest in the image."""
        if label_image is not None:
            intensity_images = [region.intensity_image for region in regions]
            imgs = [region.image for region in regions]
            data_dict = [skew(intensity[seg], axis=0, bias=True) for intensity, seg in zip(intensity_images, imgs)]
        else:
            data_dict= skew(intensity_image.reshape(-1),axis=0, bias=True)
        logger.debug('Completed extracting skewness for ' + int_file_name)
        return data_dict

    def entropy(*args):
        """Calculate entropy for all the regions of interest in the image."""
        if label_image is not None:
            intensity_images = [region.intensity_image for region in regions]
            imgs = [region.image for region in regions]
            data_dict = [shannon_entropy(intensity[seg]) for intensity, seg in zip(intensity_images, imgs)]
        else:
            data_dict = shannon_entropy(intensity_image.reshape(-1))
        logger.debug('Completed extracting entropy for ' + int_file_name)
        return data_dict

    def kurtosis(*args):
        """Calculate kurtosis for all the regions of interest in the image."""
        if label_image is not None:
            intensity_images = [region.intensity_image for region in regions]
            imgs = [region.image for region in regions]
            data_dict = [kurto(intensity[seg], axis=0, fisher=False, bias=True) for intensity, seg in zip(intensity_images, imgs)]
        else:
            data_dict= kurto(intensity_image.reshape(-1),axis=0, fisher=False, bias=True)
        logger.debug('Completed extracting kurtosis for ' + int_file_name)
        return data_dict

    def neighbors(seg_img, *args):
        """Calculate neighbors for all the regions of interest in the image."""
        neighbor_array = neighbors_find(seg_img, pixelDistance)
        neighbor_list = neighbor_array.tolist()
        neighbor = neighbor_list
        logger.debug('Completed extracting neighbors for ' + seg_file_names1.name)
        return neighbor

    def all(seg_img, units, int_img):
        """Calculate all features for all the regions of interest in the image."""
        #calculate area
        all_volume = volume(seg_img, units)
        #calculate neighbors
        all_neighbor = neighbors(seg_img)
        #calculate solidity
        all_solidity = solidity(seg_img)
        #calculate maxferet
        all_maxferet = maxferet(seg_img, units)
        #calculate convex area
        all_convex = convex_volume(seg_img, units)
        #calculate orientation
        all_orientation = orientation(seg_img)
        #calculate centroid row value
        all_centroidx = centroid_x(seg_img)
        #calculate centroid column value
        all_centroidy = centroid_y(seg_img)
        #calculate asphericity
        all_asphericity = asphericity(seg_img)
        #calculate acylindricity
        all_acylindricity = acylindricity(seg_img)
        #calculate anisotrophy
        all_anisotrophy = anisotrophy(seg_img)
        #calculate minimum principal moment
        all_minimum_principal_moment = minimum_principal_moment(seg_img)
        #calculate equivalent diameter
        all_equivalent_diameter = equivalent_diameter(seg_img, units)
        #calculate euler number
        all_euler_number = euler_number(seg_img)
        #calculate major axis length
        all_major_axis_length = major_axis_length(seg_img, units)
        #calculate minor axis length
        all_minor_axis_length = minor_axis_length(seg_img, units)
        #calculate solidity
        all_solidity = solidity(seg_img)
        #calculate mean intensity
        all_mean_intensity =  mean_intensity(seg_img, int_img)
        #calculate maximum intensity value
        all_max_intensity = max_intensity(seg_img, int_img)
        #calculate minimum intensity value
        all_min_intensity = min_intensity(seg_img, int_img)
        #calculate median
        all_median = median(seg_img, int_img)
        #calculate mode
        all_mode = mode(seg_img, int_img)
        #calculate standard deviation
        all_sd = standard_deviation(seg_img, int_img)
        #calculate skewness
        all_skewness= skewness(seg_img, int_img)
        #calculate entropy
        all_entropy= entropy(seg_img, int_img)
        #calculate kurtosis
        all_kurtosis = kurtosis(seg_img, int_img)
        logger.debug('Completed extracting all features for ' + seg_file_names1.name)
        return (all_volume, all_acylindricity, all_anisotrophy, all_asphericity, all_centroidx, all_centroidy, all_convex, all_entropy, all_equivalent_diameter, all_euler_number, all_kurtosis, all_major_axis_length, all_maxferet, all_max_intensity, all_mean_intensity, all_median, all_min_intensity,all_minimum_principal_moment, all_minor_axis_length, all_mode, all_neighbor, all_orientation, all_sd, all_skewness, all_solidity)

    #Dictionary of input features
    FEAT = {'volume': volume,
            'orientation': orientation,
            'convex_volume': convex_volume,
            'centroid_x': centroid_x,
            'centroid_y': centroid_y,
            'equivalent_diameter': equivalent_diameter,
            'euler_number': euler_number,
            'major_axis_length': major_axis_length,
            'minor_axis_length': minor_axis_length,
            'solidity': solidity,
            'mean_intensity': mean_intensity,
            'max_intensity': max_intensity,
            'min_intensity': min_intensity,
            'median': median,
            'mode': mode,
            'standard_deviation': standard_deviation,
            'skewness': skewness,
            'entropy': entropy,
            'kurtosis': kurtosis,
            'neighbors': neighbors,
            'maxferet': maxferet,
            'asphericity': asphericity,
            'acylindricity': acylindricity,
            'anisotrophy': anisotrophy,
            'minimum_principal_moment':minimum_principal_moment,
            'all': all}
    
    if label_image is not None:
        #Calculate features given as input for all images
        regions = measure.regionprops(label_image, intensity_image)
        #Remove the cells touching the border
        cleared = clear_border(label_image)
        #pass the filename in csv
        title = seg_file_names1.name
    if label_image is None:
        title = int_file_name

    #If features parameter left empty then raise value error 
    if not features:
        raise ValueError('Select features for extraction.')
    
    #Check whether the pixels per unit contain values when embeddedpixelsize is not required and metric for unitlength is entered
    if unitLength and not embeddedpixelsize:
        if not pixelsPerunit:
            raise ValueError('Enter pixels per unit value.')
    if label_image is not None:        
        logger.info('Extracting features for ' + seg_file_names1.name)
    else:
        logger.info('Extracting features for ' + int_file_name)

    #Centroid values shown separately in output as centroid_x and centroid_y
    if 'centroid' in features:
        features.remove('centroid')
        features.append('centroid_x')
        features.append('centroid_y')

    for each_feature in features:
        #Dynamically call the function based on the features required
        if label_image is not None:
            feature_value = FEAT[each_feature](label_image,unitLength,intensity_image)
        else:
            feature_value = FEAT[each_feature](intensity_image)
        #get all features
        if each_feature  == 'all':
            #create dataframe for all features
            df=pd.DataFrame(feature_value)
            df = df.T

            # Change the units depending on selection
            if embeddedpixelsize:
                units = img_emb_unit
            elif unitLength and not embeddedpixelsize:
                units = unitLength
            else:
                units = "voxels"

            columns = [
                f'volume_{units}',
                'acylindricity',
                'anisotrophy',
                'asphericity',
                'centroid_x',
                'centroid_y',
                f'convex_volume_{units}',
                'entropy',
                f'equivalent_diameter_{units}',
                'euler_number',
                'kurtosis',
                f'major_axis_length_{units}',
                f'maxferet_{units}',
                'maximum_intensity',
                'mean_intensity',
                'median',
                'minimum_intensity',
                'minimum_principal_moment',
                f'minor_axis_length_{units}',
                'mode',
                'neighbors',
                'orientation',
                'standard_deviation',
                'skewness',
                'solidity'
             ]
            df.columns = [c+f'' for c in columns]
            if unitLength and not embeddedpixelsize:
                check_cols = [col for col in df.columns if 'volume' in col]
                df.columns = [x + '^3' if x in check_cols else x for x in df]
            if channel is None:
                df.columns = [c+f'' for c in df.columns]
            else:
                df.columns = [c+f'_channel{channel}' for c in df.columns]
            df.columns = map(str.lower, df.columns)
        else:
            #create dataframe for features selected
            if label_image is None:
                df = pd.DataFrame({each_feature: feature_value},index=[0])
            else:
                df = pd.DataFrame({each_feature: feature_value})
            if 'volume' or 'convex_volume' or 'equivalent_diameter' or 'major_axis_length' or 'maxferet' or 'minor_axis_length' in df.columns:
                # Change the units depending on selection
                if embeddedpixelsize:
                    units = img_emb_unit
                elif unitLength and not embeddedpixelsize:
                    units = unitLength
                else:
                    units = "voxels"

                df.rename({
                    "volume": f"volume_{units}",
                    "convex_volume": f"convex_volume_{units}",
                    "equivalent_diameter": f"equivalent_diameter_{units}", 
                    "major_axis_length": f"major_axis_length_{units}", 
                    "minor_axis_length": f"minor_axis_length_{units}",
                    "maxferet": f"maxferet_{units}"
                }, axis='columns', inplace=True)
                columns = [c+f'' for c in df.columns]
                if unitLength and not embeddedpixelsize:
                    check_cols = [col for col in df.columns if 'volume' in col]
                    if check_cols:
                        df.columns = [col+'^3'for col in check_cols]
                if channel is None:
                    df.columns = [c+f'' for c in df.columns]
                else:
                    df.columns = [c+f'_channel{channel}' for c in df.columns]
                df.columns = map(str.lower, df.columns)
        df_insert = pd.concat([df_insert, df], axis=1)

 
    if label_image is not None:
        #Lists all the labels in the image
        label = [r.label for r in regions]
        
        if len(label)==1:
            df_insert.insert(0, 'mask_image', title)
            if intensity_image is not None:
               df_insert.insert(1, 'intensity_image', int_file_name) 
        else:
            #Measure region props for only the object not touching the border
            regions1 = np.unique(cleared)[1:]

            #List of labels for only objects that are not touching the border
            label_nt_touching = regions1-1
            #Find whether the object is touching border or not 
            border_cells = np.full((len(regions)),True,dtype=bool)       
            border_cells[label_nt_touching]=False
            if intensity_image is None:
            #Create column label and image
                data = { 'label': label,
                         'mask_image':title}                     
                data1 = {'touching_border': border_cells}
                df1 = pd.DataFrame(data,columns=['label','mask_image'])
                df_values= ['label','mask_image']
            else:
                data = { 'label': label,
                         'mask_image':title,
                         'intensity_image':int_file_name}                     
                data1 = {'touching_border': border_cells}
                df1 = pd.DataFrame(data,columns=['label','mask_image','intensity_image'])
                df_values= ['label','mask_image','intensity_image']
            #Create column touching border
            df2 = pd.DataFrame(data1,columns=['touching_border'])
            df_insert1 = pd.concat([df1,df_insert,df2],ignore_index=True, axis=1)
            dfch = df_insert.columns.tolist()
            
            df_values1 = ['touching_border']
            joinedlist= df_values + dfch + df_values1
            df_insert = df_insert1
            df_insert.columns =joinedlist
    if label_image is None:
        #Insert filename as 1st column     
        df_insert.insert(0, 'intensity_image', int_file_name)
    return df_insert, title

# Setup the argument parsing
def main():
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='Everything you need to start a Feature Extraction plugin.')
    parser.add_argument('--features', dest='features', type=str,
                        help='Features to calculate', required=True)
    parser.add_argument('--filePattern', dest='filePattern', type=str,
                        help='The filepattern used to match files with each other.', required=True)
    parser.add_argument('--csvfile', dest='csvfile', type=str,
                        help='Save csv as separate or single file', required=True)
    parser.add_argument('--embeddedpixelsize', dest='embeddedpixelsize', type=str,
                        help='Embedded pixel size if present', required=False)
    parser.add_argument('--pixelsPerunit', dest='pixelsPerunit', type=float,
                        help='Pixels per unit', required= False)
    parser.add_argument('--unitLength', dest='unitLength', type=str,
                        help='Required units for features extracted', required= False)
    parser.add_argument('--intDir', dest='intDir', type=str,
                        help='Intensity image collection', required=False)
    parser.add_argument('--pixelDistance', dest='pixelDistance', type=int,
                        help='Pixel distance to calculate the neighbors touching cells', required=False)
    parser.add_argument('--segDir', dest='segDir', type=str,
                        help='Segmented image collection', required=False)
    parser.add_argument('--outDir', dest='outDir', type=str,
                        help='Output collection', required=True)

    # Parse the arguments
    args = parser.parse_args()

    #Parse the filepattern
    pattern = args.filePattern
    logger.info('filePattern = {}'.format(pattern))

    #List of features to be extracted
    features = args.features.split(',')
    logger.info('features = {}'.format(features))

    #Save the features extracted (as single file for all images or 1 file for each image) in csvfile
    csvfile = args.csvfile
    logger.info('csvfile = {}'.format(csvfile))

    #Embedded pixel size if true, get units from metadata
    embeddedpixelsize = args.embeddedpixelsize
    logger.info('embeddedpixelsize = {}'.format(embeddedpixelsize))

    #Required units for the features extracted
    unitLength = args.unitLength
    logger.info('unitLength = {}'.format(unitLength))

    #Pixels per unit vaue for the units mentined in unitLength
    pixelsPerunit = args.pixelsPerunit
    logger.info('pixels per unit = {}'.format(pixelsPerunit))

    #Path to intensity image directory
    intDir = args.intDir
    logger.info('intDir = {}'.format(intDir))

    #Pixel distance to calculate neighbors
    pixelDistance = args.pixelDistance
    logger.info('pixelDistance = {}'.format(pixelDistance))

    #Path to labeled image directory
    segDir = args.segDir
    logger.info('segDir = {}'.format(segDir))

    #Path to save output csv files
    outDir = args.outDir
    logger.info('outDir = {}'.format(outDir))

    logger.info("Started")

    df_csv = pd.DataFrame([])
    
    if not segDir and not intDir:
        raise ValueError('No input image specified.')

    intensity_features = ['mean_intensity','max_intensity','min_intensity','median','mode','skewness','kurtosis','standard_deviation','entropy']
    if intDir and not segDir:
        if 'all' in features:
            raise ValueError('No labeled/segmented image specified.')
        elif (all(fe not in intensity_features for fe in features)):
            raise ValueError('No labeled/segmented image specified.')
        elif (any(fe not in intensity_features for fe in features)):
            logger.warning('No labeled/segmented image specified.')
            features = [i for i in features if i in intensity_features]
    elif segDir and not intDir:
        if 'all' in features:
            raise ValueError('No intensity image specified.')
            features = [i for i in features if i in intensity_features]
        elif (all(fe in intensity_features for fe in features)):
            raise ValueError('No intensity image specified.')
        elif (any(fe in intensity_features for fe in features)):
            logger.warning('No intensity image specified.')
            features = [i for i in features if i not in intensity_features]
            
    #Get list of .ome.tif files in the directory including sub folders for labeled images
    # Try to infer a filepattern from the files on disk for faster matching later
    if segDir:
        configfiles_seg = filepattern.FilePattern(segDir,pattern)
        files_seg = list(configfiles_seg())
    else:
        label_image = None
    files_int=[]
    #Get list of .ome.tif files in the directory including sub folders for intensity images
    if intDir:      
        configfiles_int = filepattern.FilePattern(intDir,pattern)
        files_int = list(configfiles_int())
    else:
        intensity_image = None
        
    #Check for matching filepattern
    if segDir and intDir:
        if len(files_seg) == 0 and len(files_int)==0 :
            raise ValueError("Could not find files matching filepattern")
    elif segDir and not intDir:
        if len(files_seg) == 0:
            raise ValueError("Could not find labeled/segmented image files matching filepattern")
    elif intDir and not segDir:
        if len(files_int) == 0:
            raise ValueError("Could not find intensity image files matching filepattern")

        
    if not segDir:
        for intfile in files_int:
            df=None
            channel=None
            intensity_image,img_emb_unit = read(intfile[0]['file'])
            int_name = intfile[0]['file'].name
            df,title = feature_extraction(features,
                                          embeddedpixelsize,
                                          unitLength,
                                          pixelsPerunit,
                                          pixelDistance,
                                          channel,
                                          intensity_image,
                                          img_emb_unit,
                                          label_image=None,
                                          seg_file_names1=None,
                                          int_file_name=int_name)
            os.chdir(outDir)
            if csvfile == 'separatecsv':
                logger.info('Saving dataframe to csv for ' + intfile[0]['file'].name)
                export_csv = df.to_csv(r'%s.csv'%title, index=None, header=True, encoding='utf-8-sig')
            else:
                df_csv = df_csv.append(df)

    elif segDir:   
        #Run analysis for each labeled image in the list
        for img_file in itertools.zip_longest(files_seg,files_int):
            label_image,img_emb_unit = read(img_file[0][0]['file'])
            df = None
            files=''
            channel=''
            #Both intensity and labeled image passed as input
            if intDir:
                #Get matching files
                files = configfiles_int.get_matching(**{k.upper():v for k,v in img_file[0][0].items() if k not in ['file','c']})
                if files is not None:  
                    if len(files) == 0 and(all(fe not in intensity_features for fe in features)):
                       intensity_image=None 

                    elif len(files) == 0 and(any(fe in intensity_features for fe in features)):
                       logger.warning(f"Could not find intensity files matching label image, {img_file[0][0]['file'].name}. Skipping...")
                       if df==None:
                           continue
                    else:
                        intensity_image,img_emb_unit = read(files[0]['file'])
                        int_filename = files[0]['file'].name
                        
                    #Check length of files to mention channels in output only when there is more than one channel
                    if len(files)==1:
                        channel = None

                    for file in files:
                        if channel != None:
                            channel=file['c']
                        dfc,title = feature_extraction(features,
                                          embeddedpixelsize,
                                          unitLength,
                                          pixelsPerunit,
                                          pixelDistance,
                                          channel,
                                          intensity_image,
                                          img_emb_unit,
                                          label_image,
                                          img_file[0][0]['file'],
                                          int_filename)
                        if df is None:
                            df = dfc
                        else:
                            df = pd.concat([df, dfc.iloc[:,2:]], axis=1,sort=False)

                    if csvfile == 'singlecsv':
                        df_csv = df_csv.append(df)
                    
                else:
                    if len(files_seg) != len(files_int) :
                        raise ValueError("Number of labeled/segmented images is not equal to number of intensity images")
                    #Read intensity image
                    intensity_image,img_emb_unit = read(img_file[1][0]['file'])
                    int_file = img_file[1][0]['file'].name
                    channel=None
                
            #Dataframe contains the features extracted from images 
            if not intDir or files==[] or files==None:
                channel=None
                int_filename = None
                if intDir and files==None:
                    int_filename = int_file
                df,title = feature_extraction(features,
                                          embeddedpixelsize,
                                          unitLength,
                                          pixelsPerunit,
                                          pixelDistance,
                                          channel,
                                          intensity_image,
                                          img_emb_unit,
                                          label_image,
                                          seg_file_names1=img_file[0][0]['file'],
                                          int_file_name=int_filename
                                          )
            #Save each csv file separately
            os.chdir(outDir)
            
            if csvfile == 'singlecsv' and (files ==''or files==None or files==[]):
                 df_csv = df_csv.append(df)
            elif csvfile == 'separatecsv':
                if df.empty:
                    raise ValueError('No output to save as csv files')
                else:
                    logger.info('Saving dataframe to csv for ' + img_file[0][0]['file'].name)
                    df = df.loc[:,~df.columns.duplicated()]
                    if 'touching_border' in df.columns:
                        last_column = df.pop('touching_border')
                        df.insert(len(df.columns), 'touching_border', last_column)
                    export_csv = df.to_csv(r'%s.csv'%title, index=None, header=True, encoding='utf-8-sig')


    #Save values for all images in single csv
    if csvfile == 'singlecsv':
         if df_csv.empty:
             raise ValueError('No output to save as csv files')
         else:
            logger.info('Saving dataframe to csv file for all images')
            df_csv.dropna(inplace=True, axis=1, how='all')
            df_csv = df_csv.loc[:,~df_csv.columns.duplicated()]
            if 'touching_border' in df_csv.columns:
                last_column = df_csv.pop('touching_border')
                df_csv.insert(len(df_csv.columns), 'touching_border', last_column)
            export_csv = df_csv.to_csv(r'Feature_Extraction.csv', index=None, header=True, encoding='utf-8-sig')

if __name__ == "__main__":
    main()