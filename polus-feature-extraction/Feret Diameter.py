import numpy as np
import math
from operator import itemgetter

def box_border_search(label_image,boxsize):
    """Get borders of object to calculate feret diameter memory efficiently.
    
    Parameters
    ----------
    label_image : (N,M) ndarray
        Labeled input image.
    boxsize : int
        Box size value.
    
    Returns
    -------
    border : array
        A array containing the borders of the object n.
    
    Notes
    -----
    boxsize = 2*pixel_distance+1
    Boxsize finds pixels with different labels within d pixel_distance.
    
    Examples
    --------
    >>>perimeter = box_border_search(label_image,boxsize=3)
    >>>perimeter
    [[  1   1   1 ...  21  21  21]
     [  1   0   0 ...   0   0  21]
     [  1   0   0 ...   0   0  21]
     ...
     [561   0   0 ...   0   0 545]
     [561   0   0 ...   0   0 545]
     [561 561 561 ... 545 545 545]]
    
    """

#Get image shape values
    height, width = label_image.shape
    
#Get boxsize values
    floor_offset = math.floor(boxsize/2)
    ceil_offset = math.ceil(boxsize/2)
    
#Create the integral image
    int_image = np.zeros((height+1, width+1))
    int_image[1:,1:] = np.cumsum(np.cumsum(np.double(label_image),0),1)
    int_image_transpose = int_image.T
    int_image_int = int_image_transpose.astype(int)
    
#Create indices for the original image
    height_sequence = height-(boxsize-1)
    width_sequence = width-(boxsize-1)
    width_boxsize = np.linspace(0, width-boxsize, height_sequence)
    height_boxsize = np.linspace(0, height-boxsize, width_sequence)
    columns, rows = np.meshgrid(width_boxsize, height_boxsize)
    columns_flat = columns.flatten(1)
    columns_reshape = columns_flat.reshape(-1,1)
    rows_flat = rows.flatten(1)
    rows_reshape = rows_flat.reshape(-1,1)
    upper_left = (height+1)*columns_reshape+rows_reshape
    upper_left_int = upper_left.astype(int)
    upper_right = upper_left_int+(boxsize)*(height+1)
    upper_right_int = upper_right.astype(int)
    lower_left = upper_left+boxsize
    lower_left_int = lower_left.astype(int)
    lower_right = upper_right_int+boxsize
    lower_right_int = lower_right.astype(int)
    
#Get the sum of local neighborhood defined by boxSize
    int_image_flat = int_image_int.flatten(1)
    int_image_flat_transpose = int_image_flat.T
    neighborvals = (int_image_flat_transpose[upper_left_int]
                    + int_image_flat_transpose[lower_right_int] 
                    - int_image_flat_transpose[upper_right_int] 
                    - int_image_flat_transpose[lower_left_int])
    
#Divide the pixel averages by the pixel value
    reshape_vals = np.reshape(neighborvals, (height-2*floor_offset, width-2*floor_offset))
    double_image = label_image[ceil_offset-1: -floor_offset,ceil_offset-1: -floor_offset]
    pix_mask = reshape_vals / double_image
    pad = np.pad(pix_mask, ((floor_offset, floor_offset), (floor_offset, floor_offset)), mode='constant')
    thresh = boxsize*boxsize
    
#Get perimeter of the object    
    pad_array = np.array(pad)
    pad_flat = pad_array.flatten(1)
    perimeter_indices = np.where(pad_flat!=thresh)
    perimeter_indices_array = np.asarray(perimeter_indices)
    perimeter_indices_reshape = perimeter_indices_array.reshape(-1,1)
    perimeter_zeros = np.zeros(label_image.shape)
    perimeter_int = perimeter_zeros.astype(int)
    perimeter_flat = perimeter_int.flatten()
    image_flat = label_image.flatten(1)
    perimeter_flat[perimeter_indices_reshape] = image_flat[perimeter_indices_reshape]
    perimeter_reshape=perimeter_flat.reshape(height, width)
    perimeter_transpose=perimeter_reshape.T
    return perimeter_transpose

def feret_diameter(label_image,theta):
    """Calculate the feret diameter of object n.
    
    Parameters
    ----------
    label_image : (N,M) ndarray
        Labeled input image.
    theta : range()
        Angles at which to compute the feret diameter.
    
    Returns
    -------
    feret_diameter : array
        Array contains the feret diameters of the corresponding objects at each of the angles in theta.
        
    Examples
    --------
    >>>edges= box_border_search(label_image, 3)
    >>>feretdiam = feret_diameter(edges,range(1,181))
    >>>minferet = np.min(feretdiam)
    >>>minferet
    5.0955
    
    >>>maxferet = np.max(feretdiam)
    >>>maxferet
    12.5987 
    
    """
    uniqueindices_list=[]
    meanind_list=[]
    rot_position =[]
    rot_list =[]
    sub_rot_list=[]
    feretdiam=[]
    counts_scalar_copy=None
    
#Convert to radians    
    theta = np.asarray(theta)
    theta = np.radians(theta)
    
#Get border of objects
    obj_edges = box_border_search(label_image,boxsize=3)
    
#Get indices and label of all pixels
    obj_edges_flat = obj_edges.flatten(1)
    obj_edges_reshape = obj_edges_flat.reshape(-1,1)
    objnum = obj_edges_reshape[obj_edges_reshape!=0]
    obj_edges_transpose = obj_edges.T
    positionx = np.where(obj_edges_transpose)[0]
    positionx_reshape = positionx.reshape(-1,1)
    positiony = np.where(obj_edges_transpose)[1]
    positiony_reshape = positiony.reshape(-1,1)
    index = list(range(len(objnum)))
    index = np.asarray(index).reshape(objnum.shape)
    stack_index_objnum = np.column_stack((index,objnum))
    
#Sort pixels by label
    sort_index_objnum = sorted(stack_index_objnum, key=itemgetter(1))
    index_objnum_array = np.asarray(sort_index_objnum)
    index_split = index_objnum_array[:,0]
    objnum_split = index_objnum_array[:,1]
    positionx_index = positionx_reshape[index_split]
    positiony_index = positiony_reshape[index_split]
    
#Get number of pixels for each object    
    objnum_reshape = np.asarray(objnum_split).reshape(-1,1)
    difference_objnum = np.diff(objnum_reshape,axis=0)
    stack_objnum = np.vstack((1,difference_objnum,1))
    objbounds = np.where(stack_objnum)
    objbounds_array = np.asarray(objbounds)
    objbounds_split = objbounds_array[0,:]
    objbounds_reshape = objbounds_split.reshape(-1,1)
    objbounds_counts = objbounds_reshape[1:]-objbounds_reshape[:-1]
    
#Create cell with x, y positions of each objects border
    for counts in objbounds_counts:
        counts_scalar = np.asscalar(counts)
        if counts_scalar == objbounds_counts[0]:
            uniqueindices_x = positionx_index[:counts_scalar]
            uniqueindices_y = positiony_index[:counts_scalar]
            counts_scalar_copy = counts_scalar
        if counts_scalar != objbounds_counts[0]:
            index_range = counts_scalar_copy+counts_scalar
            uniqueindices_x = positionx_index[counts_scalar_copy: index_range]
            uniqueindices_y = positiony_index[counts_scalar_copy: index_range]
            counts_scalar_copy = index_range
        uniqueindices_x_reshape = uniqueindices_x.reshape(-1,1)
        uniqueindices_y_reshape = uniqueindices_y.reshape(-1,1)
        uniqueindices_concate = np.concatenate((uniqueindices_x_reshape, uniqueindices_y_reshape),axis=1)
        uniqueindices_list.append(uniqueindices_concate)
        
#Center points based on object centroid    
    uniqueindices_array = np.asarray(uniqueindices_list)
    for indices in uniqueindices_array:
        length = indices.shape[0]
        repitations= (len(indices),2)
        sum_indices0 = np.sum(indices[:, 0])
        sum_indices1 = np.sum(indices[:, 1])
        length_indices0 =(sum_indices0/len(indices))
        length_indices1 =(sum_indices1/len(indices))
        mean_tile0 = np.tile(length_indices0, repitations)
        sub_mean0_indices = np.subtract(indices, mean_tile0)
        sub_mean0_indices = sub_mean0_indices[:,0]
        mean_tile1 = np.tile(length_indices1, repitations)
        sub_mean1_indices = np.subtract(indices, mean_tile1)
        sub_mean1_indices = sub_mean1_indices[:,1]
        meanind0_reshape = sub_mean0_indices.reshape(-1,1)
        meanind1_reshape = sub_mean1_indices.reshape(-1,1)
        meanind_concate = np.concatenate((meanind0_reshape, meanind1_reshape),axis=1)
        meanind_list.append(meanind_concate)
    center_point = np.asarray(meanind_list)
    
#Create transformation matrix
    rot_trans = np.array((np.cos(theta), -np.sin(theta)))
    rot_trans = rot_trans.T
    
#Calculate rotation positions
    for point in center_point:
        rot_position.clear()
        for rotation in rot_trans:
            rot_mul = np.multiply(rotation,point)
            rot_add = np.add(rot_mul[:,0],rot_mul[:,1])
            rot_position.append(rot_add)
        rot_array = np.asarray(rot_position)
        rot_list.append(rot_array)
        
#Get Ferets diameter    
    for rot in rot_list:
        sub_rot_list.clear()
        for rt,trans in zip(rot,rot_trans):
            sub_rot = np.subtract(np.max(rt),np.min(rt))
            sub_rot_add = np.add(sub_rot, np.sum(abs(trans)))
            sub_rot_list.append(sub_rot_add)
        convert_array = np.asarray(sub_rot_list)
        convert_reshape = convert_array.reshape(-1,1)
        feretdiam.append(convert_reshape)
    feret_diameter = np.asarray(feretdiam)
    return feret_diameter