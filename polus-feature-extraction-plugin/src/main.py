from skimage import measure
from skimage.measure import shannon_entropy
from skimage.segmentation import clear_border
from scipy.stats import skew
from scipy.stats import kurtosis as kurto
from scipy.stats import mode as modevalue
from scipy.sparse import csr_matrix
from scipy import stats
from operator import itemgetter
from bfio import BioReader
from itertools import repeat
from functools import partial
import argparse
import logging
import os
import math
import itertools
import filepattern
import concurrent
import cv2
import multiprocessing
import numpy as np
import pandas as pd

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
    br = BioReader(img_file)
    #Load only the first channel
    image_bfio = br[:,:,0:1,0,0].squeeze()
    #Get embedded units from metadata (physical size)
    img_unit = br.ps_y[1]
    logger.info('Reading file\t{}/ {}'.format(img_file.parent, img_file.name))
    return image_bfio, img_unit
    
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

def neighbors_find(lbl_img, im_label,pixeldistance):
    """Calculate the number of objects within d pixels of object n.

    Args:
        lbl_image (ndarray): Labeled image array.
        im_label (int): list of all labels in the image.
        pixeldistance (int): Pixel distance value.

    Returns:
        An array showing the number of neighbors touching the object for each object in labeled image. 

    Note:
        Number_of_Neighbors = neighbors_find(label_image, im_label, pixeldistance=None)
        Computes the number of objects within 5 pixels of each object.

    """
    nei=[]
    num_nei=[]
    shape_img = lbl_img.shape
    #Find contours
    contours = cv2.findContours((lbl_img==im_label).astype('uint8'),cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)[0]
    for contour in contours:
        ck= [(contour[i][j]) for i in range(0,len(contour)) for j in  range(0,len(contour[i]))]
        #Get borderpoints
        pxx = [x[0] for x in ck]
        pyy = [x[1] for x in ck]
        pypdd = [int(py-pixeldistance) for py in pyy]
        pxpdd = [int(px-pixeldistance) for px in pxx]
        pypd_rnn=[int(py+pixeldistance) for py in pyy]
        pxpd_rnn=[int(px+pixeldistance) for px in pxx]
        cand = [(l,k) for pypd,pypd_rn,pxpd,pxpd_rn in zip(pypdd,pypd_rnn,pxpdd,pxpd_rnn) for l in range(pypd,pypd_rn+1) for k in range(pxpd,pxpd_rn+1)]
        values=[lbl_img[l][k] for l,k in cand if(l<shape_img[0] and k<shape_img[1] and l>=0 and k>=0)]
        nei.append(values)
        #Get list of number of neighbors
        uniq_nei=np.unique(nei)
        num_nei.append(len(uniq_nei)-2)
        nei=[]
    num_nei = num_nei[0]
    return num_nei

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

    if neighbors < 3 or perimeter == 0:
        poly_size_ratio = "NAN"
        poly_area_ratio = "NAN"
        poly_ave = "NAN"
        hex_size_ratio = "NAN"
        hex_area_ratio = "NAN"
        hex_ave = "NAN"
        hex_sd = "NAN"
    return(poly_ave, hex_ave, hex_sd)

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
    boxsize = 3
    thetastart = 1
    thetastop = 180
    if pixelDistance is None:
        pixelDistance = 5
        
    def area(seg_img, units, *args):
        """Calculate area for all the regions of interest in the image."""          
        data_dict1 = [region.area for region in regions]
        if unitLength and not embeddedpixelsize:
            data_dict = [dt_pixel / pixelsPerunit**2 for dt_pixel in data_dict1]
        else:
            data_dict = data_dict1
        logger.debug('Completed extracting area for ' + seg_file_names1.name)
        return data_dict

    def perimeter(seg_img, units, *args):
        """Calculate perimeter for all the regions of interest in the image."""
        data_dict1 = [region.perimeter for region in regions]
        if unitLength and not embeddedpixelsize:
            data_dict = [dt_pixel / pixelsPerunit for dt_pixel in data_dict1]
        else:
            data_dict = data_dict1
        logger.debug('Completed extracting perimeter for ' + seg_file_names1.name)
        return data_dict

    def orientation(seg_img,*args):
        """Calculate orientation for all the regions of interest in the image."""
        label = [region.label for region in regions]
        data_dict=[]
        def compute_M(data):
            cols = np.arange(data.size)
            data[data>=len(label)+1] = len(label)   # Limit it
            return csr_matrix((cols, (data.ravel(), cols)),shape=(len(label) + 1, data.size))
        def get_indices_sparse(data):
            M = compute_M(data)
            return [np.unravel_index(row.data, data.shape) for row in M]
        ori_data = get_indices_sparse(seg_img)
        data_pro = ori_data[1:]
        for i in data_pro:
            x=i[0]
            y=i[1]
            xg, yg = x.mean(), y.mean()
            x = x - xg
            y = y - yg
            uyy = (y**2).sum()
            uxx = (x**2).sum()
            uxy = (x*y).sum()
            if (uyy > uxx):
                num = uyy - uxx + np.sqrt((uyy - uxx)**2 + 4*uxy**2)
                den = 2*uxy
            else:
                num = 2*uxy
                den = uxx - uyy + np.sqrt((uxx - uyy)**2 + 4*uxy**2)
        
            if (num == 0) and (den == 0):
                orientation1 = 0
            else:
                value = num/den
                orientation1 = -(180/math.pi) * math.atan(value)
            data_dict.append(orientation1)
        logger.debug('Completed extracting orientation for ' + seg_file_names1.name)
        return data_dict

    def convex_area(seg_img, units, *args):
        """Calculate convex_area for all the regions of interest in the image.""" 
        data_dict1 = [region.convex_area for region in regions]
        if unitLength and not embeddedpixelsize:
            data_dict = [dt_pixel / pixelsPerunit**2 for dt_pixel in data_dict1]
        else:
            data_dict = data_dict1
        logger.debug('Completed extracting convex area for ' + seg_file_names1.name)
        return data_dict
    
    def bbox_ymin(*args):
        """Calculate bounding box xmin for all the regions of interest in the image."""
        bbox_value = [str(region.bbox) for region in regions]
        bbox_all = [value.split(',') for value in bbox_value]
        data_dict = [bbox_min[0].replace('(','') for bbox_min in bbox_all]
        logger.debug('Completed extracting boundingbox_ymin for ' + seg_file_names1.name)
        return data_dict
    
    def bbox_xmin(*args):
        """Calculate bounding box ymin for all the regions of interest in the image."""
        bbox_value = [str(region.bbox) for region in regions]
        bbox_all = [value.split(',') for value in bbox_value]
        data_dict = [bbox_min[1] for bbox_min in bbox_all]
        logger.debug('Completed extracting boundingbox_xmin for ' + seg_file_names1.name)
        return data_dict
    
    def bbox_width(*args):
        """Calculate bounding box width for all the regions of interest in the image."""
        imgs= [region.image for region in regions]
        data_dict = [w.shape[1] for w in imgs]
        logger.debug('Completed extracting boundingbox_width for ' + seg_file_names1.name)
        return data_dict
    
    def bbox_height(*args):
        """Calculate bounding box height for all the regions of interest in the image."""
        imgs= [region.image for region in regions]
        data_dict = [h.shape[0] for h in imgs]
        logger.debug('Completed extracting boundingbox_height for ' + seg_file_names1.name)
        return data_dict

    def centroid_y(*args):
        """Calculate centroidy for all the regions of interest in the image."""
        centroid_value = [str(region.centroid) for region in regions]
        cent_y= [cent.split(',') for cent in centroid_value]
        data_dict = [centroid_y[0].replace('(','') for centroid_y in cent_y]
        logger.debug('Completed extracting centroid_row for ' + seg_file_names1.name)
        return data_dict

    def centroid_x(*args):
        """Calculate centroidx for all the regions of interest in the image."""
        centroid_value = [str(region.centroid) for region in regions]
        cent_x = [cent.split(',') for cent in centroid_value]
        data_dict = [centroid_x[1].replace(')','') for centroid_x in cent_x]
        logger.debug('Completed extracting centroid_column for ' + seg_file_names1.name)
        return data_dict

    def eccentricity(*args):
        """Calculate eccentricity for all the regions of interest in the image."""
        data_dict = [region.eccentricity for region in regions]
        logger.debug('Completed extracting eccentricity for ' + seg_file_names1.name)
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

    def solidity(*args):
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
        data_dict=[]
        label=[region.label for region in regions]
        executor = concurrent.futures.ThreadPoolExecutor(max_workers = multiprocessing.cpu_count())
        results = executor.map(neighbors_find, repeat(seg_img), label,repeat(pixelDistance))
        data_dict = list(results)
        logger.debug('Completed extraction neighbors for ' + seg_file_names1.name)
        return data_dict

    def maxferet(seg_img, *args):
        """Calculate maxferet for all the regions of interest in the image."""
        edges= box_border_search(seg_img, boxsize)
        feretdiam = feret_diameter(edges, boxsize, thetastart, thetastop)
        maxferet1 = [np.max(feret) for feret in feretdiam]
        if unitLength and not embeddedpixelsize:
            maxferet = [dt_pixel / pixelsPerunit for dt_pixel in maxferet1]
        else:
            maxferet = maxferet1
        logger.debug('Completed extracting maxferet for ' + seg_file_names1.name)
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
        logger.debug('Completed extracting minferet for ' + seg_file_names1.name)
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
        if features == 'all':
            poly_hex = poly_hex_score(seg_img, units)
        poly_hex = poly_hex_score(seg_img, units)
        polygonality_score = [poly[0] for poly in poly_hex]
        logger.debug('Completed extracting polygonality score for ' + seg_file_names1.name)
        return polygonality_score

    def hexagonality_score(seg_img, units, *args):
        """Get hexagonality score for all the regions of interest in the image."""
        poly_hex = poly_hex_score(seg_img, units)
        hexagonality_score = [poly[1] for poly in poly_hex]
        logger.debug('Completed extracting hexagonality score for ' + seg_file_names1.name)
        return hexagonality_score

    def hexagonality_sd(seg_img, units, *args):
        """Get hexagonality standard deviation for all the regions of interest in the image."""
        poly_hex = poly_hex_score(seg_img, units)
        hexagonality_sd = [poly[2] for poly in poly_hex]
        logger.debug('Completed extracting hexagonality standard deviation for ' + seg_file_names1.name)
        return hexagonality_sd
    
    def all(seg_img, units, int_img):
        """Calculate all features for all the regions of interest in the image."""
        #calculate area
        all_area = area(seg_img, units)
        #calculate perimeter
        all_peri = perimeter(seg_img, units)
        #calculate neighbors
        all_neighbor = neighbors(seg_img)
        #calculate maxferet
        edges= box_border_search(seg_img, boxsize)
        feretdiam = feret_diameter(edges, boxsize, thetastart, thetastop)
        all_maxferet = [np.max(feret) for feret in feretdiam]
        #calculate minferet
        all_minferet = [np.min(feret) for feret in feretdiam]  
        if unitLength and not embeddedpixelsize:
            maxferet = [dt_pixel / pixelsPerunit for dt_pixel in all_maxferet]
            minferet = [dt_pixel / pixelsPerunit for dt_pixel in all_minferet]
        else:
            minferet = all_minferet
            maxferet = all_maxferet
        #calculate convex area
        all_convex = convex_area(seg_img, units)
        #calculate solidity
        all_solidity = np.array(all_area)/np.array(all_convex)
        #calculate orientation
        all_orientation = orientation(seg_img)
        #calculate centroid row value
        all_centroidx = centroid_x(seg_img)
        #calculate centroid column value
        all_centroidy = centroid_y(seg_img)
        #Calculate bounding box xmin
        all_bboxxmin = bbox_xmin(seg_img)
        #Calculate bounding box ymin
        all_bboxymin = bbox_ymin(seg_img)
        #Calculate bounding box width
        all_bboxwidth = bbox_width(seg_img)
        #Calculate bounding box height
        all_bboxheight = bbox_height(seg_img)
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
        #calculate polygonality_score
        all_polygon_score = [polygonality_hexagonality(area_metric, perimeter_metric, int(neighbor_metric), solidity_metric, maxferet_metric, minferet_metric) for area_metric, perimeter_metric, neighbor_metric, solidity_metric, maxferet_metric, minferet_metric in zip(all_area, all_peri, all_neighbor, all_solidity, all_maxferet, all_minferet)]#seg_img, units)
        all_polygonality_score = [poly[0] for poly in all_polygon_score]
        #calculate hexagonality_score
        all_hexagonality_score = [poly[1] for poly in all_polygon_score]
        #calculate hexagonality standarddeviation
        all_hexagonality_sd = [poly[2] for poly in all_polygon_score]
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
        logger.debug('Completed extracting all features for ' + seg_file_names1.name)
        return (all_area, all_centroidx, all_centroidy, all_bboxxmin, all_bboxymin, all_bboxwidth, all_bboxheight, all_major_axis_length, all_minor_axis_length, all_eccentricity, all_orientation, all_convex, all_euler_number, all_equivalent_diameter, all_solidity, all_peri, all_maxferet, all_minferet, all_neighbor, all_polygonality_score, all_hexagonality_score, all_hexagonality_sd, all_kurtosis, all_max_intensity, all_mean_intensity, all_median, all_min_intensity, all_mode, all_sd, all_skewness)

    #Dictionary of input features
    FEAT = {'area': area,
            'bbox_xmin': bbox_xmin,
            'bbox_ymin': bbox_ymin,
            'bbox_width': bbox_width,
            'bbox_height': bbox_height,
            'perimeter': perimeter,
            'orientation': orientation,
            'convex_area': convex_area,
            'centroid_x': centroid_x,
            'centroid_y': centroid_y,
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
    #Bounding box location values shown separately in output as bbox_xmin and bbox_ymin
    if 'boundingbox_location' in features:
        features.remove('boundingbox_location')
        features.append('bbox_xmin')
        features.append('bbox_ymin')
    #Bounding box dimension values shown separately in output as bbox_width and bbox_height
    if 'boundingbox_dimension' in features:
        features.remove('boundingbox_dimension')
        features.append('bbox_width')
        features.append('bbox_height')
    
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
                units = "pixels"

            columns = [
                f'area_{units}',
                'centroid_x',
                'centroid_y',
                'bbox_xmin',
                'bbox_ymin',
                'bbox_width',
                'bbox_height',
                f'major_axis_length_{units}', 
                f'minor_axis_length_{units}',
                'eccentricity',
                'orientation',
                f'convex_area_{units}',
                'euler_number',
                f'equivalent_diameter_{units}',
                'solidity',
                f'perimeter_{units}',
                f'maxferet_{units}',
                f'minferet_{units}',
                'neighbors',
                'polygonality_score',
                'hexagonality_score',
                'hexagonality_sd',
                'kurtosis',
                'maximum_intensity',
                'mean_intensity',
                'median',
                'minimum_intensity',
                'mode',
                'standard_deviation',
                'skewness',

            ]
            df.columns = [c+f'' for c in columns]
            if unitLength and not embeddedpixelsize:
                check_cols = [col for col in df.columns if 'area' in col]
                df.columns = [x + '^2' if x in check_cols else x for x in df]
            #Show channel values only when there is more than 1 channel
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
            if any({'area', 'convex_area', 'equivalent_diameter', 'major_axis_length', 'maxferet', 'minor_axis_length', 'minferet', 'perimeter'}.intersection (df.columns)):
                #Change the units depending on selection
                if embeddedpixelsize:
                    units = img_emb_unit
                elif unitLength and not embeddedpixelsize:
                    units = unitLength
                else:
                    units = "pixels"

                df.rename({
                    "area": f"area_{units}",
                    "convex_area": f"convex_area_{units}",
                    "equivalent_diameter": f"equivalent_diameter_{units}", 
                    "major_axis_length": f"major_axis_length_{units}", 
                    "minor_axis_length": f"minor_axis_length_{units}",
                    "maxferet": f"maxferet_{units}", 
                    "minferet": f"minferet_{units}",
                    "perimeter": f"perimeter_{units}"
                }, axis='columns', inplace=True)
                columns = [c+f'' for c in df.columns]
                if unitLength and not embeddedpixelsize:
                    check_cols = [col for col in df.columns if 'Area' in col]
                    if check_cols:
                        df.columns = [col+'^2'for col in check_cols]
                if channel is None:
                    df.columns = [c+f'' for c in df.columns]
                else:
                    df.columns = [c+f'_channel{channel}' for c in df.columns]
                df.columns = map(str.lower, df.columns)
        df_insert = pd.concat([df_insert, df], axis=1)
    
    if label_image is not None:
        #Lists all the labels in the image
        label = [r.label for r in regions]

        #Measure region props for only the object not touching the border
        regions1 = np.unique(cleared)[1:]
        #List of labels for only objects that are not touching the border
        label_nt_touching = regions1-1
        #Find whether the object is touching border or not 
        border_cells = np.full((len(regions)),True,dtype=bool)  
        label_nt_touching[label_nt_touching>=len(border_cells)] = len(border_cells)-1   # Limit it 
        border_cells[label_nt_touching]=False
        if intensity_image is None:
        #Create column label and image
            data = { 'mask_image':title,
                        'label': label}                     
            data1 = {'touching_border': border_cells}
            df1 = pd.DataFrame(data,columns=['mask_image','label'])
            df_values= ['mask_image','label']
        else:
            data = { 'mask_image':title,
                        'intensity_image':int_file_name,
                        'label': label}                     
            data1 = {'touching_border': border_cells}
            df1 = pd.DataFrame(data,columns=['mask_image','intensity_image','label'])
            df_values= ['mask_image','intensity_image','label']
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

def labeling_is_blank(label_image):
    """Check if the label image is trivial (blank, missing, non-informative).
    
    Args:
        label_image (ndarray): Labeled image array.
        
    Returns:
        True if the labeling is non-informative
        
    """
    return (label_image.min()==0 and label_image.max()==0)

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
    
    #Only intensity image as input    
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

            #Skip feature calculation and saving results for an image having trivial/blank/missing segmentation
            if labeling_is_blank(label_image):
                continue;

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
            logger.info('Saving dataframe to csv file for all images in {}'.format(outDir))
            df_csv.dropna(inplace=True, axis=1, how='all')
            df_csv = df_csv.loc[:,~df_csv.columns.duplicated()]
            if 'touching_border' in df_csv.columns:
                last_column = df_csv.pop('touching_border')
                df_csv.insert(len(df_csv.columns), 'touching_border', last_column)
            export_csv = df_csv.to_csv(r'Feature_Extraction.csv', index=None, header=True, encoding='utf-8-sig')

if __name__ == "__main__":
    main()
