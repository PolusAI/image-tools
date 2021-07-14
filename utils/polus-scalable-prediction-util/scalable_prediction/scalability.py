
import os
import shutil
import tempfile

import itertools
from itertools import repeat

from concurrent import futures
from multiprocessing import cpu_count

import bfio
from bfio import BioReader, BioWriter

import numpy as np

from csbdeep.utils import normalize
from splinedist.models import Config2D, SplineDist2D, SplineDistData2D
from splinedist.utils import phi_generator, grid_generator

import matplotlib.pyplot as plt

def get_dim1dim2(dim1 : int, 
                 image_size : int, 
                 window_size : int):
    """ This function returns the range for the dimension 
    starting at dim1 and ending at dim2.
    Args :
        dim1 : the starting point of the range
        image_size : the largest value that dim2 could be
        window_size : the typical size of the range
    Returns:
        dim1 : returns the starting point of a range
        dim2 : return the ending point of a range
    """

    # the starting and ending must be integers
    dim1 = int(dim1)

    # if at the end, then y2 would be the extents of image_size
    if dim1+window_size > image_size:
        dim2 = int(image_size)
    else:
        # otherwise it would be the (dim1 + window_size)
        dim2 = int(dim1+window_size)
    
    return (int(dim1), int(dim2))


def get_ranges(image_shape : tuple,
               window_size : tuple,
               step_size : tuple):

    """ This function generates a list of yxczt dimensions to iterate 
    through so that bfio objects can be read in tiles. 
    Args:
        image_shape : shape of the whole bfio object
        tile_len : the size of tiles
    Returns:
        yxzct_ranges : a list of dimensions to iterate through. 
                       [((y1, y2), (x1, x2), (z1, z2), (c1, c2), (t1, t2)), ... ,
                        ((y1, y2), (x1, x2), (z1, z2), (c1, c2), (t1, t2))]
    """

    y_range = list(np.arange(0, image_shape[0], step_size[0]))
    y_range = list(map(get_dim1dim2, 
                       y_range, repeat(image_shape[0]), repeat(window_size[0])))

    x_range = list(np.arange(0, image_shape[1], step_size[1]))
    x_range = list(map(get_dim1dim2, 
                       x_range, repeat(image_shape[1]), repeat(window_size[1])))

    z_range = list(np.arange(0, image_shape[2], step_size[2]))
    z_range = list(map(get_dim1dim2, 
                       z_range, repeat(image_shape[2]), repeat(window_size[2])))

    c_range = list(np.arange(0, image_shape[3], step_size[3]))
    c_range = list(map(get_dim1dim2, 
                       c_range, repeat(image_shape[3]), repeat(window_size[3])))

    t_range = list(np.arange(0, image_shape[4], step_size[4]))
    t_range = list(map(get_dim1dim2, 
                       t_range, repeat(image_shape[4]), repeat(window_size[4])))

    # https://docs.python.org/3/library/itertools.html#itertools.product
    yxzct_ranges = itertools.product(y_range,x_range,z_range,c_range,t_range)
    return yxzct_ranges

def sliding_window(image : bfio.bfio.BioReader, 
                   image_size : tuple, 
                   window_size : tuple, 
                   step_size : tuple):
    """ This is a python generator that yields sections of 
    the input image so that it can be analyzed it tiles rather 
    than all at once. 
    Args:
        image : the input bfio object that is being read.
        image_size : the extents of the bfio object.
        window_size : the size of the tile
        step_size : the size of step in each dimension.
    Return:
        y1, y2 - the range of the y dimension of the bfio object
        x1, x2 - the range of the x dimension of the bfio object
        z1, z2 - the range of the z dimension of the bfio object
        c1, c2 - the range of the c dimension of the bfio object
        t1, t2 - the range of the t dimension of the bfio object
        image[y1:y2, x1:x2, z1:z2, c1:c2, t1:t2] - section of the input bfio object
    """
    yxczt_ranges = get_ranges(image_shape = image_size,
                              window_size = window_size,
                              step_size = step_size)
    
    for yxzct in yxczt_ranges:
        # return all dimension ranges (dim1 - dim2) and the tiled image 
        yield (yxzct, image[yxzct[0][0]:yxzct[0][1], 
                            yxzct[1][0]:yxzct[1][1], 
                            yxzct[2][0]:yxzct[2][1], 
                            yxzct[3][0]:yxzct[3][1],
                            yxzct[4][0]:yxzct[4][1]])

def compare_overlaps(overlap_pairs : np.ndarray):
    """ This function looks at all the overlapping pairs. 
    The first variable in the tuple is the old label and the 
    second variable in the tuple is the new label. 

    There are two dictionaries outputted so the algorithm can
    compare which new labels map to the old label, and which old labels 
    map to a new labels.  There can be overlap between the two 
    dictionaries. 

    Args:
        overlap_pairs - all the labeled segments that overlap
                        each other.
    Returns:
         overlap_dict_l2r - a dictionary showing all the new labels mapping 
                            to an old label
         overlap_dict_r2l - a dictionary showing all the old labels mapping 
                            to a new label

    """
    # initialize the two dictionaries 
    # dictionary mapping right to left, or the old label to new label of the same segment
    overlap_dict_r2l = {}
    # dictionary mapping left to right, or tje new label to old label of the same segment
    overlap_dict_l2r = {}

    for pair_i, pair_j in overlap_pairs:
    # pair_i is the old label
    # pair_j is the new label of the same segment

        # key is the old label
        # value are all the new labels
        if pair_i in overlap_dict_l2r.keys():
            overlap_dict_l2r[pair_i].append(pair_j)
        else:
            overlap_dict_l2r[pair_i] = [pair_j]

        # key is the new label
        # value are all the old labels
        if pair_j in overlap_dict_r2l.keys():
            overlap_dict_r2l[pair_j].append(pair_i)
        else:
            overlap_dict_r2l[pair_j] = [pair_i]

    return overlap_dict_l2r, overlap_dict_r2l

def scalable_prediction(bioreader_obj : bfio.bfio.BioReader,
                        biowriter_obj : bfio.bfio.BioWriter,
                        biowriter_obj_location : str,
                        overlap_size : tuple,
                        prediction_fxn,
                        step_size = (1024,1024,1,1,1)):

    """
    Args:
        bioreader_obj - bfio object that contains intensity-based information 
                       that the algorithm predicts on.
        biowriter_obj - empty bfio object that we are pasting labelled information
                      into.
        biowriter_obj_location - location of biowriter object for when the algorithm 
                                 needs to read the biowriter obj.
        overlap_size - the amount of overlap that occurs between every tile
        prediction - a function that makes the prediction on every tile
        step_size - the amount that every tile slides over by.
                    For bfio objects, all dimensions of step size must be in 
                    multiples of 1024.
    """

    assertion_string = """The number of dimensions for step_size ({})
    and overlap_size ({}) do not match,
    which are defined as {} and {}, respectively""".format(len(step_size),
                                                           len(overlap_size),
                                                           step_size,
                                                           overlap_size)
    assert len(step_size) == len(overlap_size), assertion_string

    assert isinstance(bioreader_obj, bfio.bfio.BioReader), \
        """bioreader_obj is not of <class bfio.bfio.BioReader>"""
    assert isinstance(biowriter_obj, bfio.bfio.BioWriter), \
        """biowriter_obj is not of <class bfio.bfio.BioWriter>"""

    assert_stepsize_dims  = np.array([shape%1024 if shape != 1 else 0 for shape in step_size])
    assert np.all((assert_stepsize_dims==0))

    assert_biowriter_shape = np.array([shape%1024 if shape != 1 else 0 for shape in biowriter_obj.shape])
    assert np.all((assert_biowriter_shape==0))
    
    window_size = tuple(sum(win) for win in zip(step_size, overlap_size))
    


    max_label = 0
    segment_locations = {}
    skipped_segments = []
    

    for yxzct, tiled_image in sliding_window(bioreader_obj, 
                                             bioreader_obj.shape, 
                                             window_size, 
                                             step_size):

        
        y1, y2 = yxzct[0]
        x1, x2 = yxzct[1]
        z1, z2 = yxzct[2]
        c1, c2 = yxzct[3]
        t1, t2 = yxzct[4]

        tiled_pred = prediction_fxn(tiled_image)
        tiled_pred_shape = tiled_pred.shape
        padding_to_pred = [0 if tiled_pred_shape[i] > step_size[i]
                             else abs(tiled_pred_shape[i] - step_size[i])
                             for i in range(len(tiled_pred_shape))]

        tiled_pred = np.pad(tiled_pred, ((0,padding_to_pred[0]), 
                                         (0,padding_to_pred[1]), 
                                         (0,padding_to_pred[2]),
                                         (0,padding_to_pred[3]),
                                         (0,padding_to_pred[4])), 
                                         'constant', constant_values = '0')

        # Need to get the bit that is overlapping
        if y1 == 0 and x1 == 0:
            # if its the first prediction, then no overlap exists.
            out_view = np.zeros(tiled_pred_shape)
        else:
            with bfio.BioReader(biowriter_obj_location, max_workers=1) as read_bw:
                out_view = read_bw[y1:y2, x1:x2, z1:z2, c1:c2, t1:t2]
                out_view = np.reshape(out_view, tiled_pred_shape)
                out_view = np.pad(out_view, ((0,padding_to_pred[0]), 
                                             (0,padding_to_pred[1]), 
                                             (0,padding_to_pred[2]),
                                             (0,padding_to_pred[3]),
                                             (0,padding_to_pred[4])), 
                                             'constant', constant_values = '0')
        
        # Gets all the pairs that overlap each other in the overlap strip.
        unique_pairs = np.unique(np.c_[out_view.flatten(), tiled_pred.flatten()], axis=0)
        
        # Get pairs where there is overlap between pre-existing labels and new labels. 
        # Replace new label with old label. Note: We cannot loop through this normally.
        # We must collect masks first, then do the replacement.
        overlap_pairs = unique_pairs[(unique_pairs > 0).all(axis=1), :]

        # convert the overlap_pairs to a dictionary that groups 
        # together old_labels that are split into two new segments 
        old_maps2_new, new_maps2_old = compare_overlaps(overlap_pairs)

        # Go through every new label that is identified and replace it with 
        # the 'old' label.  The old label usually has a smaller value, and is
        # more time consuming to relabel, therefore, the algorithm changes the 
        # labels in the current tile.
        for new_lab in new_maps2_old.keys():
            old_labs = new_maps2_old[new_lab]
            # if the new label only overlaps with one old label, 
            # then continue to relabel to old label
            if len(old_labs) == 1: 
                old_lab = old_labs[0]
                tiled_pred[tiled_pred == new_lab] = old_lab
                # Necessary if old_lab was mapped to more than once
                if old_lab in skipped_segments and old_lab != mini:
                    skipped_segments.remove(old_lab)
                    skipped_segments = list(np.unique(skipped_segments))
            # otherwise if the new label overlaps with more than one 
            # old label, then relabel all bits of the overlaping segments
            # with the smallest label among them. 
            else:
                mini = min(old_labs + [new_lab])
                tiled_pred[tiled_pred==new_lab] = mini
                for old_lab in old_labs:
                    if old_lab != mini:
                        out_view[out_view == old_lab] = mini
                        # check to make sure that the old label does not exist
                        # anywhere else.
                        for ypos1, xpos1 in segment_locations[old_lab]:
                            # if it does exist someplace else, then update that tile with 
                            # the new label
                            with bfio.BioReader(biowriter_obj_location, max_workers=1) as read_bw:
                                ypos1, ypos2 = get_dim1dim2(dim1=ypos1,
                                                            image_size=read_bw.shape[0],
                                                            window_size=window_size[0])
                                xpos1, xpos2 = get_dim1dim2(dim1=xpos1, 
                                                            image_size=read_bw.shape[1], 
                                                            window_size=window_size[1])
                                replace_image = read_bw[ypos1:ypos2,xpos1:xpos2]
                                replace_image[replace_image == old_label] = new_label
                                biowriter_obj[ypos1:ypos2, xpos1:xpos2] = replace_image
                            segment_locations[old_lab].remove((ypos1, xpos1))
                        # if labels are being replaced, then the algorithm needs to make sure 
                        # that the next label prediction uses that label number instead of 
                        # skipping over it.
                        if old_lab not in skipped_segments:
                            skipped_segments.append(old_lab)

        # Get new prediction labels. If it's greater than max_label, then we need to 
        # make sure they are continuous. 

        # we can ignore all the labels that are smaller than current max_label.  
        # since some of the new labels have been converted to smaller, older vlaues, then 
        # are only concerned with the numbers that are greater than the max_label from the 
        # previous tile
        new_labels = np.unique(tiled_pred)
        new_labels = new_labels[new_labels > max_label]
        new_labels_len = len(new_labels)
        num_skipped_segs = len(skipped_segments)
        skipped_segments = list(np.unique(skipped_segments))

        if new_labels_len > 0:
            
            reindexed_labels = np.arange(max_label + 1, max_label + new_labels_len + 1)
            # replace reindexed labels with segments that have been removed
            i = 0
            while ((i < new_labels_len) and (i < num_skipped_segs)):
                reindexed_labels = reindexed_labels[:-1]
                reindexed_labels = np.append(reindexed_labels, skipped_segments.pop()).astype(np.float64)
                i += 1
            
            # update the old values with the new values
            for old_label, new_label in zip(new_labels, reindexed_labels):
                tiled_pred[tiled_pred == old_label] = new_label

            # need to update the max_label with this tile's max label value
            if reindexed_labels.max() > max_label:
                max_label = reindexed_labels.max()

        new_labels = np.unique(tiled_pred)
        new_labels = new_labels[new_labels!=0]
        
        # update the segment_locations to include all the new predictions that 
        # are saved to the global output.
        for new in new_labels:
            if new in segment_locations.keys(): 
                segment_locations[new].append((y1,x1))
            else: 
                segment_locations[new] = [(y1,x1)]
        

        # some areas of overlap had no prediction and one prediction, so
        # in areas where there are no prediction (pixels values = 0) and 
        # in areas where are predictions (pixel values > 0), then the 
        # algorithm saves all instances of predictions made to the output.
        maxes = np.maximum(out_view, tiled_pred)
        biowriter_obj[y1:y2+padding_to_pred[0], 
                      x1:x2+padding_to_pred[1], 
                      z1:z2+padding_to_pred[2],
                      c1:c2+padding_to_pred[3],
                      t1:t2+padding_to_pred[4]] = maxes