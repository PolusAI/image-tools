# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 20:42:45 2019

@author: nagarajanj2
"""
import javabridge as jutil
import bioformats
import os
import numpy as np
import pandas as pd
from skimage import measure
from skimage.measure import label
from scipy.stats import skew
from scipy.stats import kurtosis as kurto
from scipy.stats import mode as mod
from scipy import stats
from operator import itemgetter
import math
import bfio.bfio as bfio
from bfio.bfio import BioReader
import difflib
import fnmatch


class ConvertImage(object):
    """
    This class reads .ome.tif files from the directory.
    
    Attributes:
        segment_dir: Path to segmented images.
        intensity_dir: Path to intensity images.
    
    """
    def __init__(self,segDir, intDir):
        """Inits ConvertImage with segment and intensity image."""
        self.segment_dir = segDir #segment image
        self.intensity_dir = intDir #intensity image
        
        
    def labeling(self, seg_file):
        """Label the object in the image."""
        all_labels = label(seg_file)
        return all_labels
        
    def convert_tiled_tiff(self,features,csvfile,outDir,pixelDistance):
        """ Read .ome.tif files."""
        df_feature = pd.DataFrame([])
        df = pd.DataFrame([])
        intensity_image = None

        #Start the java vm for using bioformats        
        jutil.start_vm(class_path=bioformats.JARS)
        
        index=0#use index for knowing the images processed
        
        #If intensity image is not passed as a parameter
        if self.intensity_dir is None:
            #Get list of .ome.tif files in the directory including sub folders for segmented images
            configfiles = [os.path.join(dirpath, seg_file_name)
            for dirpath, dirnames, files in os.walk(self.segment_dir)
            for seg_file_name in fnmatch.filter(files, '*.ome.tif')]
            
            for segfile in configfiles:#run analysis for each segmented image in the list
                seg_file=os.path.normpath(segfile)
                segfilename = seg_file.split('/')#split to get only the filename
                seg_file_names1 = segfilename[-1]
                              
                #Read the image using bioreader from bfio
                br_seg = BioReader(seg_file)
                segment_bfio = br_seg.read_image()
                seg_file = np.squeeze(segment_bfio)#squeeze the 5D array
                label_image= self.labeling(seg_file)#call labeling function to label the objects in the segmented images
                
                #Call the feature_extraction function in Analysis class
                analysis = Analysis(label_image,features,seg_file_names1, csvfile,outDir,pixelDistance,intensity_image)
                df = analysis.feature_extraction()
                
                #Check whether csvfile is csvone to save the features extracted from all the images in same csv file
                if csvfile == 'csvmany':
                    df_feature = df#assign the dataframe to save as separate file                    
                else:
                    df_feature = df_feature.append(df)#append the dataframe to save all in one csv file
                index+=1
                print("Number of images processed-",index)
                
        #If intensity image is passed as a parameter        
        else:
            #Get list of .ome.tif files in the directory including sub folders for segmented images
            configfiles = [os.path.join(dirpath, seg_filename)
            for dirpath, dirnames, files in os.walk(self.segment_dir)
            for seg_filename in fnmatch.filter(files, '*.ome.tif')]
            
            #Get list of .ome.tif files in the directory including sub folders for intensity images
            configfiles_int = [os.path.join(dirpath, seg_filename)
            for dirpath, dirnames, files in os.walk(self.intensity_dir)
            for seg_filename in fnmatch.filter(files, '*.ome.tif')]
            
            for segfile in configfiles:
    
                seg_file=os.path.normpath(segfile)
                segfilename = seg_file.split('/')#split to get only the filename
                seg_file_names1 = segfilename[-1]
            
            #for seg_file_names1 in seg_filenames1:#run analysis for each segmented image in the list
                intensity =difflib.get_close_matches(seg_file_names1, configfiles_int,n=1, cutoff=0.1)#match the filename in segmented image to the  list of intensity image filenames to match the files
                intensity_file = str(intensity[0])#get the filename of intensity image that has closest match
                
                #Read the intensity image using bioreader from bfio
                br_int = BioReader(intensity_file)
                intensity_bfio = br_int.read_image()
                intensity_image= np.squeeze(intensity_bfio)
                
                #Read the segmented image using bioreader from bfio
                #seg_file= self.segment_dir + "/" + seg_file_names1#set the entire path for the bioreader to read the image
                br_seg= BioReader(seg_file)
                segment_bfio = br_seg.read_image()
                seg_file = np.squeeze(segment_bfio)
                label_image= self.labeling(seg_file)#labels the objects
                
                #Call the feature_extraction function in Analysis class
                analysis = Analysis(label_image,features,seg_file_names1, csvfile, outDir, pixelDistance, intensity_image)
                df = analysis.feature_extraction()
                
                #Check whether csvfile is csvone to save the features extracted from all the images in same csv file
                if csvfile == 'csvmany':
                    df_feature = df
                else:
                    df_feature = df_feature.append(df)
                index+=1
                print("Number of images processed-",index)


        jutil.kill_vm()#kill the vm
        return(df_feature,seg_file_names1)#return dataframe and the segment image filename
        
      
class Analysis(ConvertImage):
    """
    This class extracts shape and intensity based features from the images.
    
    Attributes:
        boxsize: A constant value to get the perimeter pixels for calculating neighbors and feret diameter.
        thetastart: A constant value to set the starting angle range for calculting feret diameter.
        thatastop: A constant value to set the ending angle range for calculting feret diameter.
        pixeldistance: An integer value for distance between pixels to calculate the neighbors touching the object.
        features: A string indicating the feature to be extracted.
        csv_file: Option to save the csv file.
        output_dir: Path to save csv file.
        intensity_image: Intensity image array.
        label_image: Labeled image array.
        filenames: Filename of the segmented image.
   
    """
    
    def __init__(self,label_image,features, seg_file_names1, csvfile, outDir, pixelDistance, intensity_image=None):
        """Inits Analysis with boxsize, thetastart, thetastop, pixel distance, csvfile, output directory."""
        self.objneighbors = []
        self.numneighbors = []
        self.labels = []
        self.uniqueindices_list=[]
        self.meanind_list=[]
        self.rot_position =[]
        self.rot_list =[]
        self.sub_rot_list=[]
        self.feretdiam=[]
        self.area_list=[]
        self.perim_list=[]
        self.df_insert = pd.DataFrame([])
        self.df_csv = pd.DataFrame([])
        self.boxsize = 3 #box size to get the perimeter for calculating neighbors and feret diameter
        self.thetastart = 1
        self.thetastop =  180
        self.pixeldistance = pixelDistance
        if self.pixeldistance is None:
            self.pixeldistance = 5
        else:
            self.pixeldistance = pixelDistance
        self.feature = features# list of features to calculate
        self.csv_file = csvfile#save the features(as single file for all images or 1 file for each image) in csvfile
        self.output_dir = outDir#directory to save output
        self.label_image = label_image#assign labeled image
        self.intensity_image = intensity_image#assign intensity image
        del label_image,intensity_image
        self.filenames = seg_file_names1#assign filenames
        
    def box_border_search(self, label_image, boxsize):
        """Get perimeter pixels for calculating neighbors and feret diameter."""
        #Get image shape values
        height, width = label_image.shape

        #Get boxsize values
        floor_offset = math.floor(self.boxsize/2)
        ceil_offset = math.ceil(self.boxsize/2)

        #Create the integral image
        int_image = np.zeros((height+1, width+1))
        int_image[1:,1:] = np.cumsum(np.cumsum(np.double(label_image),0),1)
        int_image_transpose = int_image.T
        int_image_int = int_image_transpose.astype(int)
        del int_image,int_image_transpose

        #Create indices for the original image
        height_sequence = height-(self.boxsize-1)
        width_sequence = width-(self.boxsize-1)
        width_boxsize = np.linspace(0, width-self.boxsize, height_sequence)
        height_boxsize = np.linspace(0, height-self.boxsize, width_sequence)
        columns, rows = np.meshgrid(width_boxsize, height_boxsize)
        columns_flat = columns.flatten(order = 'F')
        columns_reshape = columns_flat.reshape(-1,1)
        rows_flat = rows.flatten(order = 'F')
        rows_reshape = rows_flat.reshape(-1,1)
        upper_left = (height+1)*columns_reshape+rows_reshape
        upper_left_int = upper_left.astype(int)
        upper_right = upper_left_int+(self.boxsize)*(height+1)
        upper_right_int = upper_right.astype(int)
        lower_left = upper_left+self.boxsize
        lower_left_int = lower_left.astype(int)
        lower_right = upper_right_int+self.boxsize
        lower_right_int = lower_right.astype(int)
        del height_sequence,width_sequence,width_boxsize,height_boxsize,columns,columns_flat,rows,rows_flat,columns_reshape,rows_reshape,upper_right,lower_left,upper_left,lower_right
    
        #Get the sum of local neighborhood defined by boxSize
        int_image_flat = int_image_int.flatten(order = 'F')
        int_image_flat_transpose = int_image_flat.T
        neighborvals = (int_image_flat_transpose[upper_left_int]
                        + int_image_flat_transpose[lower_right_int] 
                        - int_image_flat_transpose[upper_right_int] 
                        - int_image_flat_transpose[lower_left_int])
        del lower_left_int,lower_right_int,upper_right_int,upper_left_int,int_image_flat_transpose,int_image_flat,int_image_int
        
        #Divide the pixel averages by the pixel value
        reshape_vals = np.reshape(neighborvals, (height-2*floor_offset, width-2*floor_offset))
        double_image = label_image[ceil_offset-1: -floor_offset,ceil_offset-1: -floor_offset]
        pix_mask = reshape_vals / double_image
        pad = np.pad(pix_mask, ((floor_offset, floor_offset), (floor_offset, floor_offset)), mode='constant')
        thresh = self.boxsize*self.boxsize
        del neighborvals,reshape_vals,ceil_offset,double_image,pix_mask,floor_offset
        
        #Get perimeter of the object    
        pad_array = np.array(pad)
        pad_flat = pad_array.flatten(order = 'F')
        perimeter_indices = np.where(pad_flat!=thresh)
        perimeter_indices_array = np.asarray(perimeter_indices)
        perimeter_indices_reshape = perimeter_indices_array.reshape(-1,1)
        perimeter_zeros = np.zeros(label_image.shape)
        perimeter_int = perimeter_zeros.astype(int)
        perimeter_flat = perimeter_int.flatten(order = 'F')
        image_flat = label_image.flatten(order = 'F')
        perimeter_flat[perimeter_indices_reshape] = image_flat[perimeter_indices_reshape]
        perimeter_reshape = perimeter_flat.reshape(height, width)
        perimeter_transpose = perimeter_reshape.T
        del pad_array,pad_flat,thresh,perimeter_indices,perimeter_indices_array,perimeter_zeros,perimeter_int,image_flat,perimeter_indices_reshape,perimeter_flat,perimeter_reshape
        return perimeter_transpose
    
    def neighbors(self,perimeter_transpose,pixelDistance):
        """Calculate neighbors touching the object."""
        obj_edges = self.box_border_search(perimeter_transpose, self.boxsize)
        #Get the height and width of the labeled image
        height,width = obj_edges.shape

        #Generate number of samples for creating numeric sequence
        num_sequence = (2*self.pixeldistance)+1    
        pixel_distance_range = np.linspace(-self.pixeldistance,self.pixeldistance,num_sequence)
        
        #Create a rectangular grid out of an array of pixel_distance_range and an array of pixel_distance_range1 values
        column_index,row_index = np.meshgrid(pixel_distance_range,pixel_distance_range)
        
        #Convert to single column vector
        column_index_transpose = column_index.T
        row_index_transpose = row_index.T
        column_index_reshape =  column_index_transpose.reshape(-1,1)
        row_index_reshape = row_index_transpose.reshape(-1,1)
        column_index_int = column_index_reshape.astype(int)
        row_index_int = row_index_reshape.astype(int)
        del column_index_transpose,row_index_transpose,column_index_reshape,row_index_reshape,row_index,column_index,pixel_distance_range
        
        #Generate pixel neighborhood reference
        neighboroffsets = column_index_int*height+row_index_int
        neighboroffsets = neighboroffsets[neighboroffsets != 0]
        neighboroffsets = neighboroffsets.reshape(-1,1)
        
        #Get inscribed image linear indices:    
        width_sequence = width-(2*self.pixeldistance)
        height_sequence = height-(2*self.pixeldistance)
        columns_range = np.linspace(self.pixeldistance,width-self.pixeldistance-1,width_sequence)
        rows_range = np.linspace(self.pixeldistance,height-self.pixeldistance-1,height_sequence)
        columns,rows = np.meshgrid(columns_range,rows_range)
        columns_flat = columns.flatten(order = 'F')
        columns_reshape = columns_flat.reshape(-1,1)
        rows_flat = rows.flatten(order = 'F')
        rows_reshape = rows_flat.reshape(-1,1)
        linear_index = height*columns_reshape+rows_reshape
        linear_index_int = linear_index.astype(int)
        del columns_flat,rows,rows_flat,linear_index,columns_reshape,rows_reshape
        
        #Consider indices that contain objects 
        image_flatten = obj_edges.flatten(order = 'F')
        mask = image_flatten[linear_index_int]>0
        linear_index_mask = linear_index_int[mask]
        linear_index_reshape = linear_index_mask.reshape(-1,1)
        #Get indices of neighbor pixels
        neighbor_index = (neighboroffsets+linear_index_reshape.T)
        
        #Get values of neighbor pixels       
        neighborvals = image_flatten[neighbor_index]
        del linear_index_int,mask,neighboroffsets,linear_index_reshape,neighbor_index   
        
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
        del image_flatten,linear_index_mask,objnum,stack_index_objnum,sort_index_objnum,index_split,objnum_split,index
        
        #Find object index boundaries
        difference_objnum = np.diff(objnum_reshape,axis=0)
        stack_objnum = np.vstack((1,difference_objnum,1))
        objbounds = np.where(stack_objnum)
        objbounds_array = np.asarray(objbounds)
        objbounds_split = objbounds_array[0,:]
        objbounds_reshape = objbounds_split.reshape(-1,1)
        del objbounds_split,objnum_reshape,difference_objnum,stack_objnum,objbounds,objbounds_array
        
        self.objneighbors = []
        #Get border objects  
        for obj in range(len(objbounds_reshape)-1):
            allvals = neighborvals[:, index_reshape[np.arange(objbounds_reshape[obj],objbounds_reshape[obj+1])]]
            sortedvals = np.sort(allvals.ravel())
            sortedvals_reshape = sortedvals.reshape(-1,1)
            difference_sortedvals = np.diff(sortedvals_reshape,axis=0)
            difference_sortedvals_flat = difference_sortedvals.flatten(order = 'C')
            difference_sortedvals_stack = np.hstack((1,difference_sortedvals_flat))
            uniqueindices = np.where(difference_sortedvals_stack)
            uniqueindices_array = np.asarray(uniqueindices)
            uniqueindices_transpose = uniqueindices_array.T
            obj_neighbor = sortedvals_reshape[uniqueindices_transpose]
            obj_neighbor_flat = obj_neighbor.flatten(order = 'C')
            self.objneighbors.append(obj_neighbor_flat)
            del obj_neighbor_flat,allvals,sortedvals,difference_sortedvals,difference_sortedvals_flat,difference_sortedvals_stack,uniqueindices,uniqueindices_array,uniqueindices_transpose,obj_neighbor
        objneighbors_array = np.asarray(self.objneighbors)
        del objbounds_reshape,neighborvals,index_reshape
        
        self.numneighbors = []
        self.objneighbors = []
        #Get the number of neighbor objects and its label
        for neigh in objneighbors_array:
            len_neighbor = len(neigh)-1
            self.numneighbors.append(len_neighbor)
        numneighbors_arr = np.asarray(self.numneighbors)
        numneighbors_array = numneighbors_arr.reshape(-1,1)
        del numneighbors_arr,neigh,objneighbors_array,self.numneighbors
        self.labels =[]
        return numneighbors_array


    def feret_diameter(self,perimeter_transpose,thetastart,thetastop):
        """Calculate feret diameter of the object."""
        counts_scalar_copy=None
        
        #Convert to radians
        theta = np.arange(self.thetastart,self.thetastop+1)
        theta = np.asarray(theta)
        theta = np.radians(theta)

        #Get border of objects
        obj_edges = self.box_border_search(perimeter_transpose, self.boxsize)

        #Get indices and label of all pixels
        obj_edges_flat = obj_edges.flatten(order = 'F')
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
        del obj_edges_flat,obj_edges_reshape,objnum,index,obj_edges,positionx,obj_edges_transpose,positiony
        
        #Sort pixels by label
        sort_index_objnum = sorted(stack_index_objnum, key=itemgetter(1))
        index_objnum_array = np.asarray(sort_index_objnum)
        index_split = index_objnum_array[:,0]
        objnum_split = index_objnum_array[:,1]
        positionx_index = positionx_reshape[index_split]
        positiony_index = positiony_reshape[index_split]
        del positiony_reshape,index_split,stack_index_objnum,sort_index_objnum,index_objnum_array,positionx_reshape
        
        #Get number of pixels for each object    
        objnum_reshape = np.asarray(objnum_split).reshape(-1,1)
        difference_objnum = np.diff(objnum_reshape,axis=0)
        stack_objnum = np.vstack((1,difference_objnum,1))
        objbounds = np.where(stack_objnum)
        objbounds_array = np.asarray(objbounds)
        objbounds_split = objbounds_array[0,:]
        objbounds_reshape = objbounds_split.reshape(-1,1)
        objbounds_counts = objbounds_reshape[1:]-objbounds_reshape[:-1]
        del objnum_split,difference_objnum,stack_objnum,objbounds,objbounds_array,objbounds_split,objnum_reshape,objbounds_reshape
         
        self.uniqueindices_list = [] # Clear#
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
            self.uniqueindices_list.append(uniqueindices_concate)
            del uniqueindices_concate,uniqueindices_x,uniqueindices_y,uniqueindices_x_reshape,uniqueindices_y_reshape
            
        #Center points based on object centroid    
        uniqueindices_array = np.asarray(self.uniqueindices_list)
        self.meanind_list = [] # Clear#
        for indices in uniqueindices_array:
            #length = indices.shape[0]
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
            self.meanind_list.append(meanind_concate)
            del meanind_concate,sum_indices0,sum_indices1,length_indices0,mean_tile0,repitations,length_indices1,indices,mean_tile1,sub_mean0_indices,sub_mean1_indices
        del uniqueindices_array    
        center_point = np.asarray(self.meanind_list)

        #Create transformation matrix
        rot_trans = np.array((np.cos(theta), -np.sin(theta)))
        rot_trans = rot_trans.T
        self.rot_list = [] # Clear#
        
        #Calculate rotation positions
        for point in center_point:
            self.rot_position.clear()
            for rotation in rot_trans:
                rot_mul = np.multiply(rotation,point)
                rot_add = np.add(rot_mul[:,0],rot_mul[:,1])#, out = aaa)
                self.rot_position.append(rot_add)
            rot_array = np.asarray(self.rot_position)
            self.rot_list.append(rot_array)
            del rot_array,rotation,rot_mul,rot_add
        self.rot_position.clear()
        del point,center_point

        self.feretdiam = []    # Clear#
        #Get Ferets diameter  
        for rot in self.rot_list:
            self.sub_rot_list.clear()
            for rt,trans in zip(rot,rot_trans):
                sub_rot = np.subtract(np.max(rt),np.min(rt))
                sub_rot_add = np.add(sub_rot, np.sum(abs(trans)))
                self.sub_rot_list.append(sub_rot_add)
                del sub_rot_add,sub_rot,trans,rt
            convert_array = np.asarray(self.sub_rot_list)
            convert_reshape = convert_array.reshape(-1,1)
            self.feretdiam.append(convert_reshape)
            del convert_reshape,convert_array
        #del self.sub_rot_list
        self.sub_rot_list.clear()
        feret_diameter = np.asarray(self.feretdiam)
        del self.feretdiam,self.rot_list,theta,rot
        return feret_diameter
    
   
    def polygonality_hexagonality(self,area, perimeter, neighbors, solidity, maxferet, minferet):
        """ Calculate polygonality score, hexagonality score and hexagonality standarddeviation of the object."""
        
        self.area_list=[]
        self.perim_list=[]
        
        #Calculate area hull
        area_hull = area/solidity

        #Calculate Perimeter hull
        perim_hull = 6*math.sqrt(area_hull/(1.5*math.sqrt(3)))

        if neighbors == 0:
            perimeter_neighbors = float("NAN")
        elif neighbors > 0:
            perimeter_neighbors = perimeter/neighbors

        #Polygonality metrics calculated based on the number of sides of the polygon
        if neighbors > 2:
            poly_size_ratio = 1-math.sqrt((1-(perimeter_neighbors/(math.sqrt((4*area)/(neighbors*(1/(math.tan(math.pi/neighbors))))))))*(1-(perimeter_neighbors/(math.sqrt((4*area)/(neighbors*(1/(math.tan(math.pi/neighbors)))))))))
            poly_area_ratio = 1-math.sqrt((1-(area/(0.25*neighbors*perimeter_neighbors*perimeter_neighbors*(1/(math.tan(math.pi/neighbors))))))*(1-(area/(0.25*neighbors*perimeter_neighbors*perimeter_neighbors*(1/(math.tan(math.pi/neighbors)))))))

            #Calculate Polygonality Score
            poly_ave = 10*(poly_size_ratio+poly_area_ratio)/2

            #Hexagonality metrics calculated based on a convex, regular, hexagon    
            apoth1 = math.sqrt(3)*perimeter/12
            apoth2 = math.sqrt(3)*maxferet/4
            apoth3 = minferet/2
            side1 = perimeter/6
            side2 = maxferet/2
            side3 = minferet/math.sqrt(3)
            side4 = perim_hull/6

            #Unique area calculations from the derived and primary measures above        
            area1 = 0.5*(3*math.sqrt(3))*side1*side1
            area2 = 0.5*(3*math.sqrt(3))*side2*side2
            area3 = 0.5*(3*math.sqrt(3))*side3*side3
            area4 = 3*side1*apoth2
            area5 = 3*side1*apoth3
            area6 = 3*side2*apoth3
            area7 = 3*side4*apoth1
            area8 = 3*side4*apoth2
            area9 = 3*side4*apoth3
            area10 = area_hull
            area11 = area
            
            #Create an array of all unique areas
            list_area=[area1, area2, area3, area4, area5, area6, area7, area8, area9, area10, area11]
            area_uniq = np.asarray(list_area,dtype= float)

            #Create an array of the ratio of all areas to eachother   
            for ib in range (0,len(area_uniq)):
                for ic in range (ib+1,len(area_uniq)):
                    area_ratio = 1-math.sqrt((1-(area_uniq[ib]/area_uniq[ic]))*(1-(area_uniq[ib]/area_uniq[ic])))
                    self.area_list.append (area_ratio)
            area_array = np.asarray(self.area_list)
            stat_value_area=stats.describe(area_array)
            del area_uniq,list_area,area_array,self.area_list

            #Create Summary statistics of all array ratios     
            area_ratio_ave = stat_value_area.mean
            area_ratio_sd = math.sqrt(stat_value_area.variance)

            #Set the hexagon area ratio equal to the average Area Ratio
            hex_area_ratio = area_ratio_ave

            # Perimeter Ratio Calculations
            # Two extra apothems are now useful                 
            apoth4 = math.sqrt(3)*perim_hull/12
            apoth5 = math.sqrt(4*area_hull/(4.5*math.sqrt(3)))

            perim1 = math.sqrt(24*area/math.sqrt(3))
            perim2 = math.sqrt(24*area_hull/math.sqrt(3))
            perim3 = perimeter
            perim4 = perim_hull
            perim5 = 3*maxferet
            perim6 = 6*minferet/math.sqrt(3)
            perim7 = 2*area/(apoth1)
            perim8 = 2*area/(apoth2)
            perim9 = 2*area/(apoth3)
            perim10 = 2*area/(apoth4)
            perim11 = 2*area/(apoth5)
            perim12 = 2*area_hull/(apoth1)
            perim13 = 2*area_hull/(apoth2)
            perim14 = 2*area_hull/(apoth3)

            #Create an array of all unique Perimeters
            list_perim=[perim1,perim2,perim3,perim4,perim5,perim6,perim7,perim8,perim9,perim10,perim11,perim12,perim13,perim14]
            perim_uniq = np.asarray(list_perim,dtype= float)
            del list_perim

            #Create an array of the ratio of all Perimeters to eachother    
            for ib in range (0,len(perim_uniq)):
                for ic in range (ib+1,len(perim_uniq)):
                    perim_ratio = 1-math.sqrt((1-(perim_uniq[ib]/perim_uniq[ic]))*(1-(perim_uniq[ib]/perim_uniq[ic])))
                    self.perim_list.append (perim_ratio)
                    del perim_ratio
            perim_array = np.asarray(self.perim_list)
            stat_value_perim=stats.describe(perim_array)
            del perim_uniq,self.perim_list,perim_array

            #Create Summary statistics of all array ratios    
            perim_ratio_ave = stat_value_perim.mean
            perim_ratio_sd = math.sqrt(stat_value_perim.variance)

            #Set the HSR equal to the average Perimeter Ratio    
            hex_size_ratio = perim_ratio_ave
            hex_sd = np.sqrt((area_ratio_sd**2+perim_ratio_sd**2)/2)

            # Calculate Hexagonality score
            hex_ave = 10*(hex_area_ratio+hex_size_ratio)/2

        if neighbors < 3:
            poly_size_ratio = float("NAN")
            poly_area_ratio = float("NAN")
            poly_ave = float("NAN")
            hex_size_ratio = float("NAN")
            hex_area_ratio = float("NAN")
            hex_ave = float("NAN")
            hex_sd=float("NAN")
 
        return(poly_ave, hex_ave,hex_sd)
    
    def feature_extraction(self):
       """Calculate shape and intensity based features.""" 
        
       def area(self,*args):
            """Calculate area."""
            data_dict = [region.area for region in regions]
            return data_dict
        
       def perimeter(self,*args):
            """Calculate perimeter."""
            data_dict = [region.perimeter for region in regions]
            return data_dict
        
       def orientation(self,*args):
            """Calculate orientation."""
            data_dict = [region.orientation for region in regions]
            return data_dict
        
       def convex_area(self,*args):
            """Calculate convex_area."""
            data_dict = [region.convex_area for region in regions]
            return data_dict
        
       def centroid_row(self,*args):
            """Calculate centroidx."""
            centroid_value = [str(region.centroid) for region in regions]
            cent_x= [cent.split(',') for cent in centroid_value]
            data_dict = [centroid_x[0].replace('(','') for centroid_x in cent_x]
            return data_dict
        
       def centroid_column(self,*args):
            """Calculate centroidy."""
            centroid_value = [str(region.centroid) for region in regions]
            cent_y = [cent.split(',') for cent in centroid_value]
            data_dict = [centroid_y[1].replace(')','') for centroid_y in cent_y]
            return data_dict
        
       def eccentricity(self,*args):
            """Calculate eccentricity."""
            data_dict = [region.eccentricity for region in regions]
            return data_dict
        
       def equivalent_diameter(self,*args):
            """Calculate equivalent_diameter."""
            data_dict = [region.equivalent_diameter for region in regions]
            return data_dict
        
       def euler_number(self,*args):
            """Calculate euler_number."""
            data_dict = [region.euler_number for region in regions]
            return data_dict
        
       def major_axis_length(self,*args):
            """Calculate major_axis_length."""
            data_dict = [region.major_axis_length for region in regions]
            return data_dict
        
       def minor_axis_length(self,*args):
            """Calculate minor_axis_length."""
            data_dict = [region.minor_axis_length for region in regions]
            return data_dict
        
       def solidity(self,*args):
            """Calculate solidity."""
            data_dict = [region.solidity for region in regions]
            return data_dict
        
       def mean_intensity(seg_img,int_img):
            """Calculate mean_intensity."""
            data_dict = [int((region.mean_intensity)) for region in regions]
            return data_dict
        
       def max_intensity(seg_img,int_img):
            """Calculate maximum intensity."""
            data_dict = [int((region.max_intensity))for region in regions]
            return data_dict
        
       def min_intensity(seg_img,int_img):
            """Calculate minimum intensity."""
            data_dict = [int((region.min_intensity))for region in regions]
            return data_dict
        
       def median(seg_img,int_img):
            """Calculate median."""
            intensity_images = [region.intensity_image for region in regions]
            imgs = [region.image for region in regions]
            data_dict = [int((np.median(intensity[seg]))) for intensity, seg in zip(intensity_images,imgs)]
            return data_dict
        
       def mode(seg_img,int_img):
            """Calculate mode."""
            intensity_images = [region.intensity_image for region in regions]
            imgs = [region.image for region in regions]
            mode_list = [mod(intensity[seg])[0] for intensity, seg in zip(intensity_images,imgs)]
            data_dict = [str(mode_ls)[1:-1] for mode_ls in mode_list]
            return data_dict
        
       def standard_deviation(seg_img,int_img):
            """Calculate standard deviation."""
            intensity_images = [region.intensity_image for region in regions]
            imgs = [region.image for region in regions]
            data_dict = [(np.std(intensity[seg])) for intensity, seg in zip(intensity_images,imgs)]
            return data_dict
        
       def skewness(seg_img,int_img):
            """Calculate skewness."""
            intensity_images = [region.intensity_image for region in regions]
            imgs = [region.image for region in regions]
            data_dict = [skew(intensity[seg], axis=0, bias=True) for intensity, seg in zip(intensity_images,imgs)]
            return data_dict
        
       def entropy(seg_img,int_img):
            """Calculate entropy."""
            intensity_images = [region.intensity_image for region in regions]
            imgs = [region.image for region in regions]
            hist_dd = [np.histogramdd(np.ravel(intensity[seg]), bins = 256)[0]/intensity[seg].size for intensity, seg in zip(intensity_images,imgs)]
            hist_greater_zero = [list(filter(lambda p: p > 0, np.ravel(h_dd))) for h_dd in hist_dd]
            data_dict = [-np.sum(np.multiply(hist_great, np.log2(hist_great))) for hist_great in hist_greater_zero]
            return data_dict
        
       def kurtosis(seg_img,int_img):
            """Calculate kurtosis."""
            intensity_images = [region.intensity_image for region in regions]
            imgs = [region.image for region in regions]
            data_dict = [kurto(intensity[seg],axis=0,fisher=False, bias=True) for intensity, seg in zip(intensity_images,imgs)]
            return data_dict
        
       def neighbors(seg_img,*args):
            """Calculate neighbors."""
            edges= self.box_border_search(seg_img, self.boxsize)
            neighbor_array = self.neighbors(edges, self.pixeldistance)
            neighbor_list = neighbor_array.tolist()
            neighbor = [str(neigh)[1:-1] for neigh in neighbor_list]
            return neighbor
        
       def maxferet(seg_img,*args):
            """Calculate maxferet."""
            edges= self.box_border_search(seg_img, self.boxsize)
            feretdiam = self.feret_diameter(edges,self.thetastart,self.thetastop)
            maxferet = [np.max(feret) for feret in feretdiam]
            return maxferet
        
       def minferet(seg_img,*args):
            """Calculate minferet."""
            edges= self.box_border_search(seg_img, self.boxsize)
            feretdiam = self.feret_diameter(edges,self.thetastart,self.thetastop)
            minferet = [np.min(feret) for feret in feretdiam]
            return minferet
        
       def polygonality_score(seg_img,*args):
            """Calculate polygonality score."""
            poly_area = area(seg_img)
            poly_peri = perimeter(seg_img)
            poly_neighbor = neighbors(seg_img)
            poly_solidity = solidity(seg_img)
            poly_maxferet = maxferet(seg_img)
            poly_minferet = minferet(seg_img)
            poly_hex= [self.polygonality_hexagonality(area_metric, perimeter_metric, int(neighbor_metric), solidity_metric, maxferet_metric, minferet_metric) for area_metric, perimeter_metric, neighbor_metric, solidity_metric, maxferet_metric, minferet_metric in zip(poly_area, poly_peri, poly_neighbor, poly_solidity, poly_maxferet, poly_minferet)]
            polygonality_score = [poly[0] for poly in poly_hex]
            return polygonality_score
        
       def hexagonality_score(seg_img,*args):
            """Calculate hexagonality score."""
            poly_area = area(seg_img)
            poly_peri = perimeter(seg_img)
            poly_neighbor = neighbors(seg_img)
            poly_solidity = solidity(seg_img)
            poly_maxferet = maxferet(seg_img)
            poly_minferet = minferet(seg_img)
            poly_hex= [self.polygonality_hexagonality(area_metric, perimeter_metric, int(neighbor_metric), solidity_metric, maxferet_metric, minferet_metric) for area_metric, perimeter_metric, neighbor_metric, solidity_metric, maxferet_metric, minferet_metric in zip(poly_area, poly_peri, poly_neighbor, poly_solidity, poly_maxferet, poly_minferet)]
            hexagonality_score = [poly[1] for poly in poly_hex]
            return hexagonality_score
        
       def hexagonality_sd(seg_img,*args):
            """Calculate hexagonality standard deviation."""
            poly_area = area(seg_img)#calculate area
            poly_peri = perimeter(seg_img)#calculate perimeter
            poly_neighbor = neighbors(seg_img)#calculate neighbors
            poly_solidity = solidity(seg_img)#calculate solidity
            poly_maxferet = maxferet(seg_img)#calculate maxferet
            poly_minferet = minferet(seg_img)#calculate minferet
            poly_hex= [self.polygonality_hexagonality(area_metric, perimeter_metric, int(neighbor_metric), solidity_metric, maxferet_metric, minferet_metric) for area_metric, perimeter_metric, neighbor_metric, solidity_metric, maxferet_metric, minferet_metric in zip(poly_area, poly_peri, poly_neighbor, poly_solidity, poly_maxferet, poly_minferet)]
            hexagonality_sd = [poly[2] for poly in poly_hex]
            return hexagonality_sd
        
       def all(seg_img,int_img):
            """Calculate all features."""
            all_area = area(seg_img)#calculate area
            all_peri = perimeter(seg_img)#calculate perimeter
            all_neighbor = neighbors(seg_img)#calculate neighbors
            all_solidity = solidity(seg_img)#calculate solidity
            all_maxferet = maxferet(seg_img)#calculate maxferet
            all_minferet = minferet(seg_img)#calculate minferet
            all_convex = convex_area(seg_img)#calculate convex area
            all_orientation = orientation(seg_img)#calculate orientation
            all_centroidx = centroid_row(seg_img)#calculate centroid row value
            all_centroidy = centroid_column(seg_img)#calculate centroid column value
            all_eccentricity = eccentricity(seg_img)#calculate eccentricity
            all_equivalent_diameter = equivalent_diameter(seg_img)#calculate equivalent diameter
            all_euler_number = euler_number(seg_img)#calculate euler number
            all_major_axis_length = major_axis_length(seg_img)#calculate major axis length
            all_minor_axis_length = minor_axis_length(seg_img)#calculate minor axis length
            all_solidity = solidity(seg_img)#calculate solidity
            all_neighbor = neighbors(seg_img)#calculate neighbors
            all_maxferet = maxferet(seg_img)#calculate maxferet
            all_minferet = minferet(seg_img)#calculate minferet
            poly_hex= [self.polygonality_hexagonality(area_metric, perimeter_metric, int(neighbor_metric), solidity_metric, maxferet_metric, minferet_metric) for area_metric, perimeter_metric, neighbor_metric, solidity_metric, maxferet_metric, minferet_metric in zip(all_area, all_peri, all_neighbor, all_solidity, all_maxferet, all_minferet)]
            all_polygonality_score = [poly[0] for poly in poly_hex]#calculate polygonality_score
            all_hexagonality_score = [poly[1] for poly in poly_hex]#calculate hexagonality_score
            all_hexagonality_sd = [poly[2] for poly in poly_hex]#calculate hexagonality standarddeviation   
            all_mean_intensity =  mean_intensity(seg_img,int_img)#calculate mean intensity
            all_max_intensity = max_intensity(seg_img,int_img)#calculate maximum intensity value
            all_min_intensity = min_intensity(seg_img,int_img)#calculate minimum intensity value
            all_median = median(seg_img,int_img)#calculate median
            all_mode = mode(seg_img,int_img)#calculate mode
            all_sd = standard_deviation(seg_img,int_img)#calculate standard deviation
            all_skewness= skewness(seg_img,int_img)#calculate skewness
            all_kurtosis = kurtosis(seg_img,int_img)#calculate kurtosis
            return (all_area,all_centroidx,all_centroidy,all_convex,all_eccentricity,all_equivalent_diameter,all_euler_number,all_hexagonality_score,all_hexagonality_sd,all_kurtosis,all_major_axis_length,all_maxferet,all_max_intensity,all_mean_intensity,all_median,all_min_intensity,all_minferet,all_minor_axis_length,all_mode,all_neighbor,all_orientation,all_peri,all_polygonality_score,all_sd,all_skewness,all_solidity)
        
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
       regions = measure.regionprops(self.label_image,self.intensity_image)#region properties
       title = self.filenames#to pass the filename in csv
        
       for each_feature in self.feature:
           label = [r.label for r in regions]#get labels list for all regions
           feature_value = FEAT[each_feature](self.label_image,self.intensity_image)#dynamically call the function based on the features required
            
            #get all features
           if each_feature  == 'all':
               df=pd.DataFrame(feature_value)#create dataframe for all features
               df = df.T#transpose
               df.columns =['Area','Centroid row','Centroid column','Convex area','Eccentricity','Equivalent diameter','Euler number','Hexagonality score','Hexagonality sd','Kurtosis','Major axis length','Maxferet','Maximum intensity','Mean intensity','Median','Minimum intensity','Minferet','Minor axis length','Mode','Neighbors','Orientation','Perimeter','Polygonality score','Standard deviation','Skewness','Solidity']
           else:    
               df = pd.DataFrame({each_feature: feature_value})#create dataframe for features selected
           self.df_insert = pd.concat([self.df_insert, df], axis=1)    
       self.df_insert.insert(0, 'Image', title)#Insert filename as 1st column 
       self.df_insert.insert(1, 'Label', label)#Insert label as 2nd column
       self.df_insert.columns = map (lambda x: x.capitalize(),self.df_insert.columns)#Capitalize the first letter of header
        
        #save each csv file separately
       if self.csv_file == 'csvmany':
           csv_file= Df_Csv_multiple(self.df_insert, self.output_dir,title)
           csv_final = csv_file.csvfilesave()
       else:
           self.df_csv = self.df_csv.append(self.df_insert)
       return self.df_csv

  
class Df_Csv_single(Analysis):
    """
    This class saves features for all images in same csv file.
    
    Attributes:
        df_export: Dataframe that should be converted to csv file.
        output_dir: Path to save csvfile.
    """
    def __init__(self,df_csv, output_dir):
        """Inits Df_csv_single with dataframe and output directory."""
        self.df_export = df_csv
        self.output_dir = output_dir
 
    def csvfilesave(self):
        """Save as csv file in output directory."""
        os.chdir(self.output_dir)
        #Export to csv
        export_csv = self.df_export.to_csv (r'Feature_Extraction.csv', index = None, header=True)
        del self.df_export,export_csv

         
class Df_Csv_multiple(Analysis):
    """
    This class saves features for each image in separate csv file.
    
    Attributes:
        df_exportsep: Dataframe that should be converted to csv file.
        output_dir: Path to save csvfile.
        title: Filename of the image.
    """
    def __init__(self,df_insert, output_dir,title):
        """Inits Df_csv_multiple with dataframe and output directory."""
        self.df_exportsep = df_insert
        self.output_dir = output_dir
        self.title = title

    def csvfilesave(self):
        """Save as csv file in output directory."""
        os.chdir(self.output_dir)
        #Export to csv
        export_csv = self.df_exportsep.to_csv (r'Feature_Extraction_%s.csv'%self.title, index = None, header=True)
        del self.df_exportsep,export_csv
