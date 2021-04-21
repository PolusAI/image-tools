from skimage import measure
from skimage.segmentation import clear_border
from scipy.stats import skew
from scipy.stats import kurtosis as kurto
from scipy.stats import mode as mod
from scipy import stats
from operator import itemgetter
from bfio.bfio import BioReader
import argparse
import logging
import os
import fnmatch
import difflib
import bioformats
import math
import javabridge as jutil
import numpy as np
import pandas as pd

# Initialize the logger
logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("main")
logger.setLevel(logging.INFO)

def list_file(img_directory):
    """List all the .ome.tif files in the directory.
    
    Args:
        img_directory (str): Path to the directory containing the input images.
        
    Returns:
        The path to directory, list of names of the subdirectories in dirpath (if any) and the filenames of .ome.tif files.
        
    """
    list_of_files = [os.path.join(dirpath, file_name)
                     for dirpath, dirnames, files in os.walk(img_directory)
                     for file_name in fnmatch.filter(files, '*.ome.tif')]
    return list_of_files

def read(img_file):
    """Read the .ome.tif image using BioReader.
    
    Args:
        img_directory (str): Path to the directory containing the input images.
        
    Returns:
        Array of the image and the embedded unit in the metadata if present else it will be none.
        
    """
    br_int = BioReader(img_file)
    #Load only the first channel
    image_bfio = br_int.read_image(C=[0])
    image_squeeze= np.squeeze(image_bfio)
    #Get embedded units from metadata (physical size)
    img_bfio_unit = br_int.physical_size_y()
    img_unit = img_bfio_unit[1]
    return image_squeeze, img_unit

def box_border_search(label_image, boxsize=3):
    """Get perimeter pixels of object for calculating neighbors and feret diameter memory efficiently.
    
    Args:
        label_image (ndarray): Labeled image array.
        boxsize (int): Box size value.
    
    Returns:
        An array containing the perimeter pixels of the object n.
        
    """
    #Get image shape values
    height, width = label_image.shape

    #Get boxsize values
    floor_offset = math.floor(boxsize / 2)
    ceil_offset = math.ceil(boxsize / 2)

    #Create the integral image
    int_image = np.zeros((height + 1, width + 1))
    int_image[1:, 1:] = np.cumsum(np.cumsum(np.double(label_image), 0), 1)
    int_image_transpose = int_image.T
    int_image_int = int_image_transpose.astype(int)
    del int_image, int_image_transpose

    #Create indices for the original image
    height_sequence = height - (boxsize - 1)
    width_sequence = width - (boxsize - 1)
    width_boxsize = np.linspace(0, width - boxsize, height_sequence)
    height_boxsize = np.linspace(0, height - boxsize, width_sequence)
    columns, rows = np.meshgrid(width_boxsize, height_boxsize)
    columns_flat = columns.flatten(order = 'F')
    columns_reshape = columns_flat.reshape(-1, 1)
    rows_flat = rows.flatten(order = 'F')
    rows_reshape = rows_flat.reshape(-1, 1)
    #Upper left value
    upper_left = (height + 1) * columns_reshape + rows_reshape
    upper_left_int = upper_left.astype(int)
    #Upper right value
    upper_right = upper_left_int + (boxsize) * (height + 1)
    upper_right_int = upper_right.astype(int)
    #Lower left value
    lower_left = upper_left + boxsize
    lower_left_int = lower_left.astype(int)
    #Lower right value
    lower_right = upper_right_int + boxsize
    lower_right_int = lower_right.astype(int)
    del height_sequence, width_sequence, width_boxsize, height_boxsize, columns, columns_flat, rows, rows_flat, columns_reshape, rows_reshape, upper_right, lower_left, upper_left, lower_right

    #Get the sum of local neighborhood defined by boxSize
    int_image_flat = int_image_int.flatten(order = 'F')
    int_image_flat_transpose = int_image_flat.T
    neighborvals = (int_image_flat_transpose[upper_left_int]
                    + int_image_flat_transpose[lower_right_int] 
                    - int_image_flat_transpose[upper_right_int] 
                    - int_image_flat_transpose[lower_left_int])
    del lower_left_int, lower_right_int, upper_right_int, upper_left_int, int_image_flat_transpose, int_image_flat, int_image_int
    
    #Divide the pixel averages by the pixel value
    reshape_vals = np.reshape(neighborvals, (height - 2 * floor_offset, width - 2 * floor_offset))
    double_image = label_image[ceil_offset - 1: -floor_offset, ceil_offset - 1: -floor_offset]
    pix_mask = reshape_vals / double_image
    pad = np.pad(pix_mask, ((floor_offset, floor_offset), (floor_offset, floor_offset)), mode='constant')
    thresh = boxsize * boxsize
    del neighborvals, reshape_vals, ceil_offset, double_image, pix_mask, floor_offset
    
    #Get perimeter of the object    
    pad_array = np.array(pad)
    pad_flat = pad_array.flatten(order = 'F')
    #Get perimeter indices
    perimeter_indices = np.where(pad_flat != thresh)
    perimeter_indices_array = np.asarray(perimeter_indices)
    perimeter_indices_reshape = perimeter_indices_array.reshape(-1, 1)
    perimeter_zeros = np.zeros(label_image.shape)
    perimeter_int = perimeter_zeros.astype(int)
    perimeter_flat = perimeter_int.flatten(order = 'F')
    image_flat = label_image.flatten(order = 'F')
    #Calculate perimeter
    perimeter_flat[perimeter_indices_reshape] = image_flat[perimeter_indices_reshape]
    perimeter_reshape = perimeter_flat.reshape(height, width)
    perimeter_transpose = perimeter_reshape.T
    del pad_array, pad_flat, thresh, perimeter_indices, perimeter_indices_array, perimeter_zeros, perimeter_int, image_flat, perimeter_indices_reshape, perimeter_flat, perimeter_reshape
    return perimeter_transpose

def neighbors_find(lbl_img, boxsize, pixeldistance):
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
    #Get perimeter pixels   
    obj_edges = box_border_search(lbl_img, boxsize=3)
    
    #Get the height and width of the labeled image
    height,width = obj_edges.shape

    #Generate number of samples for creating numeric sequence
    num_sequence = (2 * pixeldistance) + 1    
    pixel_distance_range = np.linspace(-pixeldistance, pixeldistance, num_sequence)
    
    #Create a rectangular grid out of an array of pixel_distance_range and an array of pixel_distance_range1 values
    column_index, row_index = np.meshgrid(pixel_distance_range, pixel_distance_range)
    
    #Convert to single column vector
    column_index_transpose = column_index.T
    row_index_transpose = row_index.T
    column_index_reshape =  column_index_transpose.reshape(-1, 1)
    row_index_reshape = row_index_transpose.reshape(-1, 1)
    column_index_int = column_index_reshape.astype(int)
    row_index_int = row_index_reshape.astype(int)
    del column_index_transpose, row_index_transpose, column_index_reshape, row_index_reshape, row_index, column_index, pixel_distance_range
    
    #Generate pixel neighborhood reference
    neighboroffsets = column_index_int * height + row_index_int
    neighboroffsets = neighboroffsets[neighboroffsets != 0]
    neighboroffsets = neighboroffsets.reshape(-1, 1)
    
    #Get inscribed image linear indices:    
    width_sequence = width - (2 * pixeldistance)
    height_sequence = height - (2 * pixeldistance)
    columns_range = np.linspace(pixeldistance, width - pixeldistance - 1, width_sequence)
    rows_range = np.linspace(pixeldistance, height - pixeldistance - 1, height_sequence)
    columns, rows = np.meshgrid(columns_range, rows_range)
    columns_flat = columns.flatten(order = 'F')
    columns_reshape = columns_flat.reshape(-1, 1)
    rows_flat = rows.flatten(order = 'F')
    rows_reshape = rows_flat.reshape(-1, 1)
    linear_index = height * columns_reshape + rows_reshape
    linear_index_int = linear_index.astype(int)
    del columns_flat, rows, rows_flat, linear_index, columns_reshape, rows_reshape
    
    #Consider indices that contain objects 
    image_flatten = obj_edges.flatten(order = 'F')
    mask = image_flatten[linear_index_int]>0
    linear_index_mask = linear_index_int[mask]
    linear_index_reshape = linear_index_mask.reshape(-1, 1)
    
    #Get indices of neighbor pixels
    neighbor_index = (neighboroffsets + linear_index_reshape.T)
    
    #Get values of neighbor pixels       
    neighborvals = image_flatten[neighbor_index]
    del linear_index_int, mask, neighboroffsets, linear_index_reshape, neighbor_index   
    
    #Sort pixels by object    
    objnum = image_flatten[linear_index_mask]
    objnum_reshape = objnum.reshape(-1, 1)
    index = list(range(len(objnum_reshape)))
    index = np.asarray(index).reshape(objnum.shape)
    stack_index_objnum = np.column_stack((index, objnum))
    sort_index_objnum = sorted(stack_index_objnum, key = itemgetter(1))
    index_objnum_array = np.asarray(sort_index_objnum)
    index_split = index_objnum_array[:, 0]
    objnum_split = index_objnum_array[:, 1]
    index_reshape = np.asarray(index_split).reshape(-1, 1)
    objnum_reshape = np.asarray(objnum_split).reshape(-1, 1)
    del image_flatten, linear_index_mask, objnum, stack_index_objnum, sort_index_objnum, index_split, objnum_split,index
    
    #Find object index boundaries
    difference_objnum = np.diff(objnum_reshape, axis=0)
    stack_objnum = np.vstack((1, difference_objnum, 1))
    objbounds = np.where(stack_objnum)
    objbounds_array = np.asarray(objbounds)
    objbounds_split = objbounds_array[0, :]
    objbounds_reshape = objbounds_split.reshape(-1, 1)
    del objbounds_split, objnum_reshape, difference_objnum, stack_objnum, objbounds, objbounds_array
    
    objneighbors = []
    #Get border objects  
    for obj in range(len(objbounds_reshape) - 1):
        allvals = neighborvals[:, index_reshape[np.arange(objbounds_reshape[obj], objbounds_reshape[obj + 1])]]
        sortedvals = np.sort(allvals.ravel())
        sortedvals_reshape = sortedvals.reshape(-1, 1)
        difference_sortedvals = np.diff(sortedvals_reshape, axis=0)
        difference_sortedvals_flat = difference_sortedvals.flatten(order = 'C')
        difference_sortedvals_stack = np.hstack((1, difference_sortedvals_flat))
        uniqueindices = np.where(difference_sortedvals_stack)
        uniqueindices_array = np.asarray(uniqueindices)
        uniqueindices_transpose = uniqueindices_array.T
        obj_neighbor = sortedvals_reshape[uniqueindices_transpose]
        obj_neighbor_flat = obj_neighbor.flatten(order = 'C')
        objneighbors.append(obj_neighbor_flat)
        del obj_neighbor_flat, allvals, sortedvals, difference_sortedvals, difference_sortedvals_flat, difference_sortedvals_stack, uniqueindices, uniqueindices_array, uniqueindices_transpose, obj_neighbor
    objneighbors_array = np.asarray(objneighbors)
    del objbounds_reshape, neighborvals, index_reshape
    
    numneighbors = []
    objneighbors = []
    #Get the number of neighbor objects and its label
    for neigh in objneighbors_array:
        len_neighbor = len(neigh) - 1
        numneighbors.append(len_neighbor)
    numneighbors_arr = np.asarray(numneighbors)
    numneighbors_array = numneighbors_arr.reshape(-1, 1)
    return numneighbors_array

def feret_diameter(lbl_img, boxsize, thetastart, thetastop):
    """Calculate the maximum caliper diamter and minimum caliper diameter of an object at angle(1-180degrees).
    
    Args:
        lbl_image (ndarray): Labeled image array.
        boxsize (int): Box size value.
        thetastart (int): Angle start value by default it is 1.
        thetastop (int): Angle stop value by default it is 180.
    
    Returns:
        An array with feret diameters of the corresponding objects at each of the angles in theta.
        
    """
    counts_scalar_copy=None
    
    #Convert to radians
    theta = np.arange(thetastart, thetastop + 1)
    theta = np.asarray(theta)
    theta = np.radians(theta)

    #Get perimeter of objects
    obj_edges = box_border_search(lbl_img, boxsize=3)

    #Get indices and label of all pixels
    obj_edges_flat = obj_edges.flatten(order = 'F')
    obj_edges_reshape = obj_edges_flat.reshape(-1, 1)
    objnum = obj_edges_reshape[obj_edges_reshape != 0]
    obj_edges_transpose = obj_edges.T
    positionx = np.where(obj_edges_transpose)[0]
    positionx_reshape = positionx.reshape(-1, 1)
    positiony = np.where(obj_edges_transpose)[1]
    positiony_reshape = positiony.reshape(-1, 1)
    index = list(range(len(objnum)))
    index = np.asarray(index).reshape(objnum.shape)
    stack_index_objnum = np.column_stack((index, objnum))
    del obj_edges_flat, obj_edges_reshape, objnum, index, obj_edges, positionx, obj_edges_transpose, positiony
    
    #Sort pixels by label
    sort_index_objnum = sorted(stack_index_objnum, key=itemgetter(1))
    index_objnum_array = np.asarray(sort_index_objnum)
    index_split = index_objnum_array[:, 0]
    objnum_split = index_objnum_array[:, 1]
    positionx_index = positionx_reshape[index_split]
    positiony_index = positiony_reshape[index_split]
    del positiony_reshape, index_split, stack_index_objnum, sort_index_objnum, index_objnum_array, positionx_reshape
    
    #Get number of pixels for each object    
    objnum_reshape = np.asarray(objnum_split).reshape(-1, 1)
    difference_objnum = np.diff(objnum_reshape, axis=0)
    stack_objnum = np.vstack((1, difference_objnum, 1))
    objbounds = np.where(stack_objnum)
    objbounds_array = np.asarray(objbounds)
    objbounds_split = objbounds_array[0, :]
    objbounds_reshape = objbounds_split.reshape(-1, 1)
    objbounds_counts = objbounds_reshape[1:]-objbounds_reshape[:-1]
    del objnum_split, difference_objnum, stack_objnum, objbounds, objbounds_array, objbounds_split, objnum_reshape, objbounds_reshape
     
    uniqueindices_list = []
    #Create cell with x, y positions of each objects border
    for counts in objbounds_counts:
        counts_scalar = np.asscalar(counts)
        if counts_scalar == objbounds_counts[0]:
            uniqueindices_x = positionx_index[:counts_scalar]
            uniqueindices_y = positiony_index[:counts_scalar]
            counts_scalar_copy = counts_scalar
        if counts_scalar != objbounds_counts[0]:
            index_range = counts_scalar_copy + counts_scalar
            uniqueindices_x = positionx_index[counts_scalar_copy: index_range]
            uniqueindices_y = positiony_index[counts_scalar_copy: index_range]
            counts_scalar_copy = index_range
        uniqueindices_x_reshape = uniqueindices_x.reshape(-1, 1)
        uniqueindices_y_reshape = uniqueindices_y.reshape(-1, 1)
        uniqueindices_concate = np.concatenate((uniqueindices_x_reshape, uniqueindices_y_reshape), axis=1)
        uniqueindices_list.append(uniqueindices_concate)
        del uniqueindices_concate, uniqueindices_x, uniqueindices_y, uniqueindices_x_reshape, uniqueindices_y_reshape
        
    #Center points based on object centroid    
    uniqueindices_array = np.asarray(uniqueindices_list)
    meanind_list = []
    for indices in uniqueindices_array:
        repitations= (len(indices), 2)
        sum_indices0 = np.sum(indices[:, 0])
        sum_indices1 = np.sum(indices[:, 1])
        length_indices0 =(sum_indices0 / len(indices))
        length_indices1 =(sum_indices1 / len(indices))
        mean_tile0 = np.tile(length_indices0, repitations)
        sub_mean0_indices = np.subtract(indices, mean_tile0)
        sub_mean0_indices = sub_mean0_indices[:, 0]
        mean_tile1 = np.tile(length_indices1, repitations)
        sub_mean1_indices = np.subtract(indices, mean_tile1)
        sub_mean1_indices = sub_mean1_indices[:, 1]
        meanind0_reshape = sub_mean0_indices.reshape(-1, 1)
        meanind1_reshape = sub_mean1_indices.reshape(-1, 1)
        meanind_concate = np.concatenate((meanind0_reshape, meanind1_reshape), axis=1)
        meanind_list.append(meanind_concate)
        del meanind_concate, sum_indices0, sum_indices1, length_indices0, mean_tile0, repitations, length_indices1, indices, mean_tile1, sub_mean0_indices, sub_mean1_indices
    del uniqueindices_array    
    center_point = np.asarray(meanind_list)

    #Create transformation matrix
    rot_trans = np.array((np.cos(theta), -np.sin(theta)))
    rot_trans = rot_trans.T
    rot_list = []
    rot_position=[]
    sub_rot_list=[]
    
    #Calculate rotation positions
    for point in center_point:
        rot_position.clear()
        for rotation in rot_trans:
            rot_mul = np.multiply(rotation, point)
            rot_add = np.add(rot_mul[:, 0], rot_mul[:, 1])
            rot_position.append(rot_add)
        rot_array = np.asarray(rot_position)
        rot_list.append(rot_array)
        del rot_array, rotation, rot_mul, rot_add
    rot_position.clear()
    del point, center_point

    feretdiam = []
    #Get Ferets diameter  
    for rot in rot_list:
        sub_rot_list.clear()
        for rt,trans in zip(rot, rot_trans):
            sub_rot = np.subtract(np.max(rt), np.min(rt))
            sub_rot_add = np.add(sub_rot, np.sum(abs(trans)))
            sub_rot_list.append(sub_rot_add)
            del sub_rot_add, sub_rot, trans, rt
        convert_array = np.asarray(sub_rot_list)
        convert_reshape = convert_array.reshape(-1, 1)
        feretdiam.append(convert_reshape)
        del convert_reshape, convert_array
    sub_rot_list.clear()
    feret_diameter = np.asarray(feretdiam)
    del feretdiam, rot_list, theta, rot
    return feret_diameter

def polygonality_hexagonality(area, perimeter, neighbors, solidity, maxferet, minferet):
    """Calculate the polygonality score, hexagonality score and hexagonality standard deviation of object n.
    
    Args:
        area (int): Number of pixels of the region.
        perimeter (float): Perimeter of object which approximates the contour as a line through the centers of border pixels using a 4-connectivity.
        neighbors (int): Number of neighbors touching the object.
        solidity (float): Ratio of pixels in the region to pixels of the convex hull image.
        maxferet (float): Maximum caliper distance across the entire object.
        minferet (float): Minimum caliper distance across the entire object.
    
    Returns:
        The polygonality score ranges from -infinity to 10. Score 10 indicates the object shape is polygon and score -infinity indicates the object shape is not polygon.
        The hexagonality score ranges from -infinity to 10. Score 10 indicates the object shape is hexagon and score -infinity indicates the object shape is not hexagon.
        The dispersion of hexagonality_score relative to its mean.
        
    """
    area_list=[]
    perim_list=[]
    
    #Calculate area hull
    area_hull = area / solidity

    #Calculate Perimeter hull
    perim_hull = 6 * math.sqrt(area_hull / (1.5 * math.sqrt(3)))

    if neighbors == 0:
        perimeter_neighbors = float("NAN")
    elif neighbors > 0:
        perimeter_neighbors = perimeter / neighbors

    #Polygonality metrics calculated based on the number of sides of the polygon
    if neighbors > 2:
        poly_size_ratio = 1 - math.sqrt((1 - (perimeter_neighbors / (math.sqrt((4 * area) / (neighbors * (1 / (math.tan(math.pi / neighbors)))))))) * (1 -(perimeter_neighbors / (math.sqrt(( 4 * area) / (neighbors * (1 / (math.tan(math.pi / neighbors)))))))))
        poly_area_ratio = 1 - math.sqrt((1 - (area / (0.25 * neighbors * perimeter_neighbors * perimeter_neighbors * (1 / (math.tan(math.pi / neighbors)))))) * (1 - (area / (0.25 * neighbors * perimeter_neighbors * perimeter_neighbors * (1 / (math.tan(math.pi / neighbors)))))))

        #Calculate Polygonality Score
        poly_ave = 10 * (poly_size_ratio + poly_area_ratio) / 2

        #Hexagonality metrics calculated based on a convex, regular, hexagon    
        apoth1 = math.sqrt(3) * perimeter / 12
        apoth2 = math.sqrt(3) * maxferet / 4
        apoth3 = minferet / 2
        side1 = perimeter / 6
        side2 = maxferet / 2
        side3 = minferet / math.sqrt(3)
        side4 = perim_hull / 6

        #Unique area calculations from the derived and primary measures above        
        area1 = 0.5 * (3 * math.sqrt(3)) * side1 * side1
        area2 = 0.5 * (3 * math.sqrt(3)) * side2 * side2
        area3 = 0.5 * (3 * math.sqrt(3)) * side3 * side3
        area4 = 3 * side1 * apoth2
        area5 = 3 * side1 * apoth3
        area6 = 3 * side2 * apoth3
        area7 = 3 * side4 * apoth1
        area8 = 3 * side4 * apoth2
        area9 = 3 * side4 * apoth3
        area10 = area_hull
        area11 = area
        
        #Create an array of all unique areas
        list_area=[area1, area2, area3, area4, area5, area6, area7, area8, area9, area10, area11]
        area_uniq = np.asarray(list_area, dtype=float)

        #Create an array of the ratio of all areas to eachother   
        for ib in range (0, len(area_uniq)):
            for ic in range (ib + 1, len(area_uniq)):
                area_ratio = 1 - math.sqrt((1 - (area_uniq[ib] / area_uniq[ic])) * (1 - (area_uniq[ib] / area_uniq[ic])))
                area_list.append(area_ratio)
        area_array = np.asarray(area_list)
        stat_value_area = stats.describe(area_array)
        del area_uniq, list_area, area_array, area_list

        #Create Summary statistics of all array ratios     
        area_ratio_ave = stat_value_area.mean
        area_ratio_sd = math.sqrt(stat_value_area.variance)

        #Set the hexagon area ratio equal to the average Area Ratio
        hex_area_ratio = area_ratio_ave

        # Perimeter Ratio Calculations
        # Two extra apothems are now useful                 
        apoth4 = math.sqrt(3) * perim_hull / 12
        apoth5 = math.sqrt(4 * area_hull / (4.5 * math.sqrt(3)))

        perim1 = math.sqrt(24 * area / math.sqrt(3))
        perim2 = math.sqrt(24 * area_hull / math.sqrt(3))
        perim3 = perimeter
        perim4 = perim_hull
        perim5 = 3 * maxferet
        perim6 = 6 * minferet / math.sqrt(3)
        perim7 = 2 * area / (apoth1)
        perim8 = 2 * area / (apoth2)
        perim9 = 2 * area / (apoth3)
        perim10 = 2 * area / (apoth4)
        perim11 = 2 * area / (apoth5)
        perim12 = 2 * area_hull / (apoth1)
        perim13 = 2 * area_hull / (apoth2)
        perim14 = 2 * area_hull / (apoth3)

        #Create an array of all unique Perimeters
        list_perim = [perim1, perim2, perim3, perim4, perim5, perim6, perim7, perim8, perim9, perim10, perim11, perim12, perim13, perim14]
        perim_uniq = np.asarray(list_perim, dtype=float)
        del list_perim

        #Create an array of the ratio of all Perimeters to eachother    
        for ib in range (0, len(perim_uniq)):
            for ic in range (ib + 1, len(perim_uniq)):
                perim_ratio = 1 - math.sqrt((1 - (perim_uniq[ib] / perim_uniq[ic])) * (1 - (perim_uniq[ib] / perim_uniq[ic])))
                perim_list.append(perim_ratio)
                del perim_ratio
        perim_array = np.asarray(perim_list)
        stat_value_perim = stats.describe(perim_array)
        del perim_uniq, perim_list, perim_array

        #Create Summary statistics of all array ratios    
        perim_ratio_ave = stat_value_perim.mean
        perim_ratio_sd = math.sqrt(stat_value_perim.variance)

        #Set the HSR equal to the average Perimeter Ratio    
        hex_size_ratio = perim_ratio_ave
        hex_sd = np.sqrt((area_ratio_sd**2 + perim_ratio_sd**2) / 2)

        # Calculate Hexagonality score
        hex_ave = 10 * (hex_area_ratio + hex_size_ratio) / 2

    if neighbors < 3:
        poly_size_ratio = float("NAN")
        poly_area_ratio = float("NAN")
        poly_ave = float("NAN")
        hex_size_ratio = float("NAN")
        hex_area_ratio = float("NAN")
        hex_ave = float("NAN")
        hex_sd = float("NAN")
    return(poly_ave, hex_ave, hex_sd)
    
def feature_extraction(label_image, features, seg_file_names1, embeddedpixelsize, img_emb_uint, unitLength, pixelsPerunit, pixelDistance=5, intensity_image=None):
    """Calculate shape and intensity based features.
      
    Args:
        label_image (ndarray): Labeled image array.
        features (string): List of features to be extracted.
        seg_file_names1 (string): Filename of the labeled image.
        embeddedpixelsize (boolean): If true will consider units from metadata.
        img_emb_unit (string): Units embedded in the metadata.
        unitLength (string): Required units for the extracted features.
        pixelsPerunit (int): Pixels per unit for the metric mentioned in unitLength.
        intensity_image (ndarray): Intensity image array.
        pixelDistance (int): Distance between pixels to calculate the neighbors touching the object and default valus is 5.
    
    Returns:
        Dataframe containing the features extracted and the filename of the labeled image.
        
    """ 
    df_insert = pd.DataFrame([])
    boxsize = 3
    thetastart = 1
    thetastop = 180
    
    def area(seg_img, units, *args):
        """Calculate area for all the regions of interest in the image."""        
        data_dict1 = [region.area for region in regions]
        if unitLength and not embeddedpixelsize:
            data_dict = [dt_pixel / pixelsPerunit**2 for dt_pixel in data_dict1]
        else:
            data_dict = data_dict1
        logger.info('Completed extracting area for' + seg_file_names1)
        return data_dict
        
    def perimeter(seg_img, units, *args):
        """Calculate perimeter for all the regions of interest in the image."""
        
        data_dict1 = [region.perimeter for region in regions]
        if unitLength and not embeddedpixelsize:
            data_dict = [dt_pixel / pixelsPerunit for dt_pixel in data_dict1]
        else:
            data_dict = data_dict1
        logger.info('Completed extracting perimeter for' + seg_file_names1)
        return data_dict
    
    def orientation(*args):
        """Calculate orientation for all the regions of interest in the image."""
        data_dict = [region.orientation for region in regions]
        logger.info('Completed extracting orientation for' + seg_file_names1)
        return data_dict
    
    def convex_area(seg_img, units, *args):
        """Calculate convex_area for all the regions of interest in the image."""        
        data_dict1 = [region.convex_area for region in regions]
        if unitLength and not embeddedpixelsize:
            data_dict = [dt_pixel / pixelsPerunit**2 for dt_pixel in data_dict1]
        else:
            data_dict = data_dict1
        logger.info('Completed extracting convex area for' + seg_file_names1)
        return data_dict
    
    def centroid_row(*args):
        """Calculate centroidx for all the regions of interest in the image."""
        centroid_value = [str(region.centroid) for region in regions]
        cent_x= [cent.split(',') for cent in centroid_value]
        data_dict = [centroid_x[0].replace('(','') for centroid_x in cent_x]
        logger.info('Completed extracting centroid_row for' + seg_file_names1)
        return data_dict
    
    def centroid_column(*args):
        """Calculate centroidy for all the regions of interest in the image."""
        centroid_value = [str(region.centroid) for region in regions]
        cent_y = [cent.split(',') for cent in centroid_value]
        data_dict = [centroid_y[1].replace(')','') for centroid_y in cent_y]
        logger.info('Completed extracting centroid_column for' + seg_file_names1)
        return data_dict
    
    def eccentricity(*args):
        """Calculate eccentricity for all the regions of interest in the image."""
        data_dict = [region.eccentricity for region in regions]
        logger.info('Completed extracting eccentricity for' + seg_file_names1)
        return data_dict
    
    def equivalent_diameter(seg_img, units, *args):
        """Calculate equivalent_diameter for all the regions of interest in the image."""
        data_dict1 = [region.equivalent_diameter for region in regions]
        if unitLength and not embeddedpixelsize:
            data_dict = [dt_pixel / pixelsPerunit for dt_pixel in data_dict1]
        else:
            data_dict = data_dict1
        logger.info('Completed extracting equivalent diameter for' + seg_file_names1)
        return data_dict
    
    def euler_number(*args):
        """Calculate euler_number for all the regions of interest in the image."""
        data_dict = [region.euler_number for region in regions]
        logger.info('Completed extracting euler number for' + seg_file_names1)
        return data_dict
    
    def major_axis_length(seg_img, units, *args):
        """Calculate major_axis_length for all the regions of interest in the image."""
        data_dict1 = [region.major_axis_length for region in regions]
        if unitLength and not embeddedpixelsize:
            data_dict = [dt_pixel / pixelsPerunit for dt_pixel in data_dict1]
        else:
            data_dict = data_dict1
        logger.info('Completed extracting major axis length for' + seg_file_names1)
        return data_dict
    
    def minor_axis_length(seg_img, units, *args):
        """Calculate minor_axis_length for all the regions of interest in the image."""
        data_dict1 = [region.minor_axis_length for region in regions]
        if unitLength and not embeddedpixelsize:
            data_dict = [dt_pixel / pixelsPerunit for dt_pixel in data_dict1]
        else:
            data_dict = data_dict1
        logger.info('Completed extracting minor axis length for' + seg_file_names1)
        return data_dict
    
    def solidity(*args):
        """Calculate solidity for all the regions of interest in the image."""
        data_dict = [region.solidity for region in regions]
        logger.info('Completed extracting solidity for' + seg_file_names1)
        return data_dict
    
    def mean_intensity(*args):
        """Calculate mean_intensity for all the regions of interest in the image."""
        data_dict = [int((region.mean_intensity)) for region in regions]
        logger.info('Completed extracting mean intensity for' + seg_file_names1)
        return data_dict
    
    def max_intensity(*args):
        """Calculate maximum intensity for all the regions of interest in the image."""
        data_dict = [int((region.max_intensity))for region in regions]
        logger.info('Completed extracting maximum intensity for' + seg_file_names1)
        return data_dict
    
    def min_intensity(*args):
        """Calculate minimum intensity for all the regions of interest in the image."""
        data_dict = [int((region.min_intensity))for region in regions]
        logger.info('Completed extracting minimum intensity for' + seg_file_names1)
        return data_dict
    
    def median(*args):
        """Calculate median for all the regions of interest in the image."""
        intensity_images = [region.intensity_image for region in regions]
        imgs = [region.image for region in regions]
        data_dict = [int((np.median(intensity[seg]))) for intensity, seg in zip(intensity_images, imgs)]
        logger.info('Completed extracting median for' + seg_file_names1)
        return data_dict
    
    def mode(*args):
        """Calculate mode for all the regions of interest in the image."""
        intensity_images = [region.intensity_image for region in regions]
        imgs = [region.image for region in regions]
        mode_list = [mod(intensity[seg])[0] for intensity, seg in zip(intensity_images, imgs)]
        data_dict = [str(mode_ls)[1:-1] for mode_ls in mode_list]
        logger.info('Completed extracting mode for' + seg_file_names1)
        return data_dict
    
    def standard_deviation(*args):
        """Calculate standard deviation for all the regions of interest in the image."""
        intensity_images = [region.intensity_image for region in regions]
        imgs = [region.image for region in regions]
        data_dict = [(np.std(intensity[seg])) for intensity, seg in zip(intensity_images, imgs)]
        logger.info('Completed extracting standard deviation for' + seg_file_names1)
        return data_dict
    
    def skewness(*args):
        """Calculate skewness for all the regions of interest in the image."""
        intensity_images = [region.intensity_image for region in regions]
        imgs = [region.image for region in regions]
        data_dict = [skew(intensity[seg], axis=0, bias=True) for intensity, seg in zip(intensity_images, imgs)]
        logger.info('Completed extracting skewness for' + seg_file_names1)
        return data_dict
    
    def entropy(*args):
        """Calculate entropy for all the regions of interest in the image."""
        intensity_images = [region.intensity_image for region in regions]
        imgs = [region.image for region in regions]
        hist_dd = [np.histogramdd(np.ravel(intensity[seg]), bins = 256)[0] / intensity[seg].size for intensity, seg in zip(intensity_images, imgs)]
        hist_greater_zero = [list(filter(lambda p: p > 0, np.ravel(h_dd))) for h_dd in hist_dd]
        data_dict = [-np.sum(np.multiply(hist_great, np.log2(hist_great))) for hist_great in hist_greater_zero]
        logger.info('Completed extracting entropy for' + seg_file_names1)
        return data_dict
    
    def kurtosis(*args):
        """Calculate kurtosis for all the regions of interest in the image."""
        intensity_images = [region.intensity_image for region in regions]
        imgs = [region.image for region in regions]
        data_dict = [kurto(intensity[seg], axis=0, fisher=False, bias=True) for intensity, seg in zip(intensity_images, imgs)]
        logger.info('Completed extracting kurtosis for' + seg_file_names1)
        return data_dict
    
    def neighbors(seg_img, *args):
        """Calculate neighbors for all the regions of interest in the image."""
        edges= box_border_search(seg_img, boxsize)
        neighbor_array = neighbors_find(edges, boxsize, pixeldistance=5)
        neighbor_list = neighbor_array.tolist()
        neighbor = [str(neigh)[1: -1] for neigh in neighbor_list]
        logger.info('Completed extracting neighbors for' + seg_file_names1)
        return neighbor
    
    def maxferet(seg_img, *args):
        """Calculate maxferet for all the regions of interest in the image."""
        edges= box_border_search(seg_img, boxsize)
        feretdiam = feret_diameter(edges, boxsize, thetastart, thetastop)
        maxferet1 = [np.max(feret) for feret in feretdiam]
        if unitLength and not embeddedpixelsize:
            maxferet = [dt_pixel / pixelsPerunit for dt_pixel in maxferet1]
        else:
            maxferet = maxferet1
        logger.info('Completed extracting maxferet for' + seg_file_names1)
        return maxferet
    
    def minferet(seg_img, *args):
        """Calculate minferet for all the regions of interest in the image."""
        edges= box_border_search(seg_img, boxsize)
        feretdiam = feret_diameter(edges, boxsize, thetastart, thetastop)
        minferet1 = [np.min(feret) for feret in feretdiam]
        if unitLength and not embeddedpixelsize:
            minferet = [dt_pixel / pixelsPerunit for dt_pixel in minferet1]
        else:
            minferet = minferet1
        logger.info('Completed extracting minferet for' + seg_file_names1)
        return minferet
    
    def poly_hex_score(seg_img, units):
       """Calculate polygonality and hexagonality score for all the regions of interest in the image"""
       poly_area = area(seg_img, units)
       poly_peri = perimeter(seg_img, units)
       poly_neighbor = neighbors(seg_img)
       poly_solidity = solidity(seg_img)
       poly_maxferet = maxferet(seg_img, units)
       poly_minferet = minferet(seg_img, units)
       poly_hex= [polygonality_hexagonality(area_metric, perimeter_metric, int(neighbor_metric), solidity_metric, maxferet_metric, minferet_metric) for area_metric, perimeter_metric, neighbor_metric, solidity_metric, maxferet_metric, minferet_metric in zip(poly_area, poly_peri, poly_neighbor, poly_solidity, poly_maxferet, poly_minferet)]
       return poly_hex
       
    
    def polygonality_score(seg_img, units, *args):
        """Get polygonality score for all the regions of interest in the image."""
        poly_hex = poly_hex_score(seg_img, units)
        polygonality_score = [poly[0] for poly in poly_hex]
        logger.info('Completed extracting polygonality score for' + seg_file_names1)
        return polygonality_score
    
    def hexagonality_score(seg_img, units, *args):
        """Get hexagonality score for all the regions of interest in the image."""
        poly_hex = poly_hex_score(seg_img, units)
        hexagonality_score = [poly[1] for poly in poly_hex]
        logger.info('Completed extracting hexagonality score for' + seg_file_names1)
        return hexagonality_score
    
    def hexagonality_sd(seg_img, units, *args):
        """Get hexagonality standard deviation for all the regions of interest in the image."""
        poly_hex = poly_hex_score(seg_img, units)
        hexagonality_sd = [poly[2] for poly in poly_hex]
        logger.info('Completed extracting hexagonality standard deviation for' + seg_file_names1)
        return hexagonality_sd
    
    
    def all(seg_img, units, int_img):
        """Calculate all features for all the regions of interest in the image."""
        #calculate area
        all_area = area(seg_img, units)
        #calculate perimeter
        all_peri = perimeter(seg_img, units)
        #calculate neighbors
        all_neighbor = neighbors(seg_img)
        #calculate solidity
        all_solidity = solidity(seg_img)
        #calculate maxferet
        all_maxferet = maxferet(seg_img, units)
        #calculate minferet
        all_minferet = minferet(seg_img, units)
        #calculate convex area
        all_convex = convex_area(seg_img, units)
        #calculate orientation
        all_orientation = orientation(seg_img)
        #calculate centroid row value
        all_centroidx = centroid_row(seg_img)
        #calculate centroid column value
        all_centroidy = centroid_column(seg_img)
        #calculate eccentricity
        all_eccentricity = eccentricity(seg_img)
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
        poly_hex=  poly_hex_score(seg_img, units)
        #calculate polygonality_score
        all_polygonality_score = [poly[0] for poly in poly_hex]
        #calculate hexagonality_score
        all_hexagonality_score = [poly[1] for poly in poly_hex]
        #calculate hexagonality standarddeviation
        all_hexagonality_sd = [poly[2] for poly in poly_hex]
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
        #calculate kurtosis
        all_kurtosis = kurtosis(seg_img, int_img)
        logger.info('Completed extracting all features for' + seg_file_names1)
        return (all_area, all_centroidx, all_centroidy, all_convex, all_eccentricity, all_equivalent_diameter, all_euler_number, all_hexagonality_score, all_hexagonality_sd, all_kurtosis, all_major_axis_length, all_maxferet, all_max_intensity, all_mean_intensity, all_median, all_min_intensity, all_minferet, all_minor_axis_length, all_mode, all_neighbor, all_orientation, all_peri, all_polygonality_score, all_sd, all_skewness, all_solidity)
    
    #Dictionary of input features
    FEAT = {'area': area,
            'perimeter': perimeter,
            'orientation': orientation,
            'convex_area': convex_area,
            'centroid_row': centroid_row,
            'centroid_column': centroid_column,
            'eccentricity': eccentricity,
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
            'minferet': minferet,
            'polygonality_score': polygonality_score,
            'hexagonality_score': hexagonality_score,
            'hexagonality_sd': hexagonality_sd,
            'all': all}
      
    #Calculate features given as input for all images
    regions = measure.regionprops(label_image, intensity_image)
    
    #pass the filename in csv
    title = seg_file_names1
    
    #Remove the cells touching the border
    cleared = clear_border(label_image)
    
    #Measure region props for only the object not touching the border
    regions1 = measure.regionprops(cleared, intensity_image)
    
    #If features parameter left empty then raise value error 
    if not features:
       raise ValueError('Select features for extraction.')
     
    for each_feature in features:
       #Lists all the labels in the image
       label1 = [r.label for r in regions]
       label = [lb / 256 for lb in label1]
       
       #List of labels for only objects that are not touching the border
       label_nt = [nt_border.label for nt_border in regions1]
       label_nt_touching = [label_value / 256 for label_value in label_nt]
       
       #Find whether the object is touching border or not 
       label_yes = 'Yes'
       label_no = 'No'
       border_cells = []
       for element in label:
           if element in label_nt_touching:
               border_cells.append(label_no)
           else:
               border_cells.append(label_yes)
               
       #Check whether the pixels per unit contain values when embeddedpixelsize is not required and metric for unitlength is entered
       if unitLength and not embeddedpixelsize:
           if not pixelsPerunit:
               raise ValueError('Enter pixels per unit value.')
       
       logger.info('Started extracting features for' + seg_file_names1)
       
       #Dynamically call the function based on the features required
       feature_value = FEAT[each_feature](label_image,unitLength,intensity_image)

       #get all features
       if each_feature  == 'all':
           #create dataframe for all features
           df=pd.DataFrame(feature_value)
           df = df.T
           #if unit in metadata is considered
           if embeddedpixelsize:
               df.columns =[
                       'Area_%s'%img_emb_uint,
                       'Centroid row',
                       'Centroid column',
                       'Convex area_%s'%img_emb_uint,
                       'Eccentricity',
                       'Equivalent diameter_%s'%img_emb_uint,
                       'Euler number',
                       'Hexagonality score',
                       'Hexagonality sd',
                       'Kurtosis',
                       'Major axis length_%s'%img_emb_uint,
                       'Maxferet_%s'%img_emb_uint,
                       'Maximum intensity',
                       'Mean intensity',
                       'Median',
                       'Minimum intensity',
                       'Minferet_%s'%img_emb_uint,
                       'Minor axis length_%s'%img_emb_uint,
                       'Mode',
                       'Neighbors',
                       'Orientation',
                       'Perimeter_%s'%img_emb_uint,
                       'Polygonality score',
                       'Standard deviation',
                       'Skewness',
                       'Solidity']
           #if unitlength metric is considered
           elif unitLength and not embeddedpixelsize:
               df.columns =[
                       'Area_%s^2'%unitLength,
                       'Centroid row',
                       'Centroid column',
                       'Convex area_%s^2'%unitLength,
                       'Eccentricity',
                       'Equivalent diameter_%s'%unitLength,
                       'Euler number',
                       'Hexagonality score',
                       'Hexagonality sd',
                       'Kurtosis',
                       'Major axis length_%s'%unitLength,
                       'Maxferet_%s'%unitLength,
                       'Maximum intensity',
                       'Mean intensity',
                       'Median',
                       'Minimum intensity',
                       'Minferet_%s'%unitLength,
                       'Minor axis length_%s'%unitLength,
                       'Mode',
                       'Neighbors',
                       'Orientation',
                       'Perimeter_%s'%unitLength,
                       'Polygonality score',
                       'Standard deviation',
                       'Skewness',
                       'Solidity']
           #units in pixels
           else:
               df.columns =[
                       'Area_pixels',
                       'Centroid row',
                       'Centroid column',
                       'Convex area_pixels',
                       'Eccentricity',
                       'Equivalent diameter_pixels',
                       'Euler number',
                       'Hexagonality score',
                       'Hexagonality sd',
                       'Kurtosis',
                       'Major axis length_pixels',
                       'Maxferet_pixels',
                       'Maximum intensity',
                       'Mean intensity',
                       'Median',
                       'Minimum intensity',
                       'Minferet_pixels',
                       'Minor axis length_pixels',
                       'Mode',
                       'Neighbors',
                       'Orientation',
                       'Perimeter_pixels',
                       'Polygonality score',
                       'Standard deviation',
                       'Skewness',
                       'Solidity']
       
       else:
           #create dataframe for features selected
           df = pd.DataFrame({each_feature: feature_value})
           if 'Area' or 'Convex area' or 'Equivalent diameter' or 'Major axis length' or 'Maxferet' or 'Minor axis length' or 'Minferet' or 'Perimeter' in df.columns:
               if embeddedpixelsize:   
                   df.rename({
                           "area": "Area_%s"%img_emb_uint,
                           "convex_area": "Convex area_%s"%img_emb_uint,
                           "equivalent_diameter": "Equivalent diameter_%s"%img_emb_uint, 
                           "major_axis_length": "Major axis length_%s"%img_emb_uint, 
                           "minor_axis_length": "Minor axis length_%s"%img_emb_uint,
                           "maxferet": "Maxferet_%s"%img_emb_uint, 
                           "minferet": "Minferet_%s"%img_emb_uint,
                           "perimeter": "Perimeter_%s"%img_emb_uint
                   }, axis='columns', inplace=True)
               elif unitLength and not embeddedpixelsize:
                   df.rename({
                           "area": "Area_%s^2"%unitLength,
                           "convex_area": "Convex area_%s^2"%unitLength,
                           "equivalent_diameter": "Equivalent diameter_%s"%unitLength,
                           "major_axis_length": "Major axis length_%s"%unitLength,
                           "minor_axis_length": "Minor axis length_%s"%unitLength,
                           "maxferet": "Maxferet_%s"%unitLength,
                           "minferet": "Minferet_%s"%unitLength,
                           "perimeter": "Perimeter_%s"%unitLength
                   }, axis='columns', inplace=True)
               else:
                   df.rename({
                           "area": "Area_pixels",
                           "convex_area": "Convex area_pixels",
                           "equivalent_diameter": "Equivalent diameter_pixels",
                           "major_axis_length": "Major axis length_pixels",
                           "minor_axis_length": "Minor axis length_pixels",
                           "maxferet": "Maxferet_pixels",
                           "minferet": "Minferet_pixels",
                           "perimeter": "Perimeter_pixels"
                   }, axis='columns', inplace=True)
       df_insert = pd.concat([df_insert, df], axis=1)
       
    #Insert filename as 1st column     
    df_insert.insert(0, 'Image', title)
       
    #Insert label as 2nd column
    df_insert.insert(1, 'Label', label)
       
    #Insert touching border as 3rd column
    df_insert.insert(2, 'Touching border', border_cells)
      
    #Capitalize the first letter of header
    df_insert.columns = map (lambda x: x.capitalize(), df_insert.columns)
       
    return df_insert, title

# Setup the argument parsing
def main():
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='Everything you need to start a Feature Extraction plugin.')
    parser.add_argument('--features', dest='features', type=str,
                        help='Features to calculate', required=True)
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
                        help='Segmented image collection', required=True)
    parser.add_argument('--outDir', dest='outDir', type=str,
                        help='Output collection', required=True)

    # Parse the arguments
    args = parser.parse_args()
    
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
    
    try:
        #Start the java vm for using bioformats        
        jutil.start_vm(class_path=bioformats.JARS)
    
        #Get list of .ome.tif files in the directory including sub folders for labeled images
        configfiles_seg = list_file(segDir)
        
        #Check whether .ome.tif files are present in the labeled image directory
        if not configfiles_seg:
            raise ValueError('No labeled image .ome.tif files found.')
        
        #Get list of .ome.tif files in the directory including sub folders for intensity images
        if intDir:
            configfiles_int = list_file(intDir)
            #Check whether .ome.tif files are present in the intensity image directory
            if not configfiles_int:
                raise ValueError('No intensity image .ome.tif files found.')
        else:
            intensity_image = None
   
        
        #Run analysis for each labeled image in the list
        for segfile in configfiles_seg:
            #Get the full labeled image path
            seg_file = os.path.normpath(segfile)
            
            #split to get only the filename
            segfilename = os.path.split(seg_file)
            seg_file_names1 = segfilename[-1]
            
            logger.info('Started reading the file' + seg_file_names1)
            #Read the image using bioreader from bfio
            label_image, img_emb_uint = read(seg_file)
            logger.info('Finished reading the file' + seg_file_names1)
                         
            if intDir:
                
                #match the filename in labeled image to the  list of intensity image filenames to match the files
                intensity = difflib.get_close_matches(seg_file_names1, configfiles_int, n=1, cutoff=0.1)
                
                #get the filename of intensity image that has closest match
                intensity_file = str(intensity[0])
        
                #Read the intensity image using bioreader from bfio
                intensity_image, img_emb_uint = read(intensity_file)
                int_ravel = np.ravel(intensity_image)
                int_ravel1 = int_ravel / 255
                int_ravel_bool = np.array_equal(int_ravel1, int_ravel1.astype(bool))
               
                #check whether the array contains only zero
                countzero = not np.any(int_ravel) 
                if int_ravel_bool == True or countzero == True:
                    logger.warning('Intensity image ' + intensity_file + ' does not have any content' )
                    continue
            
            #Dataframe contains the features extracted from images   
            df,title = feature_extraction(label_image, features, seg_file_names1, embeddedpixelsize, img_emb_uint, unitLength, pixelsPerunit, pixelDistance, intensity_image)
            
            #Save each csv file separately
            os.chdir(outDir)
            if csvfile == 'separatecsv':
               logger.info('Saving dataframe to csv for' + seg_file_names1)
               export_csv = df.to_csv (r'Feature_Extraction_%s.csv'%title, index=None, header=True, encoding='utf-8-sig')
            else:
               df_csv = df_csv.append(df)
        #Save values for all images in single csv
        if csvfile == 'singlecsv':
            logger.info('Saving dataframe to csv file for all images')
            export_csv = df_csv.to_csv (r'Feature_Extraction.csv', index=None, header=True, encoding='utf-8-sig')
    finally:
        logger.info('Closing the javabridge')
        #kill the vm
        jutil.kill_vm()

    logger.info("Finished all processes!")

if __name__ == "__main__":
    main()
    
    
