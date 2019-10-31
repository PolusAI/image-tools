import numpy as np
from operator import itemgetter

def neighbors(label_image, distance=None):
    
    """Calculate the number of objects within d pixels of object n.
    
    Parameters
    ----------
    label_image : (N,M) ndarray
        Labeled input image.
    distance : int, optional
        Pixel distance.
    
    Returns
    -------
    Number_of_neighbors : 2D array
        An array showing the number of neighbors touching the object for each object in labeled image. 
        
    Notes
    -----
    Number_of_Neighbors = neighbors(label_image, distance = None)
        Computes the number of objects within 1 pixel of each object.
        
    Number_of_Neighbors = neighbors(label_image, distance = 5)
        Computes the number of objects within 5 pixels of each object.
    
    Examples
    --------
    >>>Number_of_Neighbors = neighbors(label_image,distance=None)
    >>>Number_of_Neighbors
    [[  1   1]
     [  1   2]
     [  1   3]
     [  1   4]
     [  1   5]
     [  1   6]
     [  1   7]
      ... ...
      ... ...
      ... ...]
      
    >>>Number_of_Neighbors = neighbors(label_image,distance=5)
    >>>Number_of_Neighbors
    [[  3   1]
     [  6   2]
     [  4   3]
     [  4   4]
     [  2   5]
     [  3   6]
     [  3   7]
      ... ...
      ... ...
      ... ...]
       
    """

    objneighbors = []
    numneighbors = []
    labels = []

#Consider pixel distance as optional argument
    if distance is None:
        pixel_distance = 1
    else:
        pixel_distance= distance

#Get the height and width of the labeled image
    height,width = label_image.shape
    
#Generate number of samples for creating numeric sequence
    num_sequence = (2*pixel_distance)+1    
    pixel_distance_range = np.linspace(-pixel_distance,pixel_distance,num_sequence)
    
#Create a rectangular grid out of an array of pixel_distance_range and an array of pixel_distance_range1 values
    column_index,row_index = np.meshgrid(pixel_distance_range,pixel_distance_range)
    
#Convert to single column vector
    column_index_transpose = column_index.T
    row_index_transpose = row_index.T
    column_index_reshape =  column_index_transpose.reshape(-1,1)
    row_index_reshape = row_index_transpose.reshape(-1,1)
    column_index_int = column_index_reshape.astype(int)
    row_index_int = row_index_reshape.astype(int)
    
#Generate pixel neighborhood reference
    neighboroffsets = column_index_int*height+row_index_int
    neighboroffsets = neighboroffsets[neighboroffsets != 0]
    neighboroffsets = neighboroffsets.reshape(-1,1)
    
#Get inscribed image linear indices:    
    width_sequence = width-(2*pixel_distance)
    height_sequence = height-(2*pixel_distance)
    columns_range = np.linspace(pixel_distance,width-pixel_distance-1,width_sequence)
    rows_range = np.linspace(pixel_distance,height-pixel_distance-1,height_sequence)
    columns,rows = np.meshgrid(columns_range,rows_range)
    columns_flat = columns.flatten(1)
    columns_reshape = columns_flat.reshape(-1,1)
    rows_flat = rows.flatten(1)
    rows_reshape = rows_flat.reshape(-1,1)
    linear_index = height*columns_reshape+rows_reshape
    linear_index_int = linear_index.astype(int)
    
#Consider indices that contain objects 
    image_flatten = label_image.flatten(1)
    mask = image_flatten[linear_index_int]>0
    linear_index_mask = linear_index_int[mask]
    linear_index_reshape = linear_index_mask.reshape(-1,1)
    
#Get indices of neighbor pixels
    neighbor_index = (neighboroffsets+linear_index_reshape.T)
    
#Get values of neighbor pixels
    neighborvals = image_flatten[neighbor_index]
    
#Sort pixels by object    
    objnum = image_flatten[linear_index_mask]
    objnum_reshape = objnum.reshape(-1,1)
    index = list(range(len(objnum_reshape)))
    index = np.asarray(index).reshape(objnum.shape)
    stack_index_objnum= np.column_stack((index,objnum))
    sort_index_objnum = sorted(stack_index_objnum, key = itemgetter(1))
    index_objnum_array = np.asarray(sort_index_objnum)
    index_split = index_objnum_array[:,0]
    objnum_split = index_objnum_array[:,1]
    index_reshape = np.asarray(index_split).reshape(-1,1)
    objnum_reshape = np.asarray(objnum_split).reshape(-1,1)
    
#Find object index boundaries
    difference_objnum = np.diff(objnum_reshape,axis=0)
    stack_objnum = np.vstack((1,difference_objnum,1))
    objbounds = np.where(stack_objnum)
    objbounds_array = np.asarray(objbounds)
    objbounds_split = objbounds_array[0,:]
    objbounds_reshape = objbounds_split.reshape(-1,1)
    
#Get border objects  
    for obj in range(len(objbounds_reshape)-1):
        allvals = neighborvals[:, index_reshape[np.arange(objbounds_reshape[obj],objbounds_reshape[obj+1])]]
        sortedvals = np.sort(allvals.ravel())
        sortedvals_reshape = sortedvals.reshape(-1,1)
        difference_sortedvals = np.diff(sortedvals_reshape,axis=0)
        difference_sortedvals_flat = difference_sortedvals.flatten()
        difference_sortedvals_stack = np.hstack((1,difference_sortedvals_flat))
        uniqueindices = np.where(difference_sortedvals_stack)
        uniqueindices_array = np.asarray(uniqueindices)
        uniqueindices_transpose = uniqueindices_array.T
        obj_neighbor = sortedvals_reshape[uniqueindices_transpose]
        obj_neighbor_flat = obj_neighbor.flatten()
        objneighbors.append(obj_neighbor_flat)
    objneighbors_array = np.asarray(objneighbors)
    
#Get the number of neighbor objects and its label
    for neigh in objneighbors_array:
        len_neighbor = len(neigh)-1
        numneighbors.append(len_neighbor)
    numneighbors_array = np.asarray(numneighbors)
    numneighbors_array = numneighbors_array.reshape(-1,1)
    for num in range(len(numneighbors_array)):
        labels.append(num+1)
    labels_array = np.asarray(labels)
    labels_reshape= labels_array.reshape(-1,1)
    neighbor_label = np.column_stack((numneighbors_array, labels_reshape))
    return neighbor_label