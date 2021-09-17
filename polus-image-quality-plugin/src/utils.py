from bfio import BioReader, BioWriter
import argparse, logging, os
import numpy as np
from pathlib import Path
import scipy
from scipy.fftpack import fft2
from scipy.ndimage.measurements import sum as nd_sum
import cv2
import csv
import imquality.brisque as brisque
import skimage
from skimage.feature import greycomatrix, greycoprops
from dom import DOM
import random

class Image_quality:

    """ Extract metrics to determine Image Quality
        Parameters
        ----------
        path : Path to the Image directory (inputDir)
        image : ndarray
        minI: int, float
        Minimum intensity to normalize image intensities
        maxI: int, float
        Maximum intensity to normalize image intensities
        scale: int, float
        Divides the the (M * N) image in to the non overlapping tiles for the given scale 
        Returns
        -------
        Output CSV 
        Notes
        -------
        Image normalization minimum and maximum intensity ranges are following
        1) 0 , 1
        2) -1 , 1
        3) 0 , 255
        """
    def __init__(self, path, image, minI, maxI, scale):
        self.path = path
        self.image = skimage.color.rgb2gray(image)
        self.minI = minI
        self.maxI = maxI
        self.scale = scale

    def normalize_intensities(self):
        assert self.minI in [0, -1], 'Invalid minI value!!'
        assert self.maxI in [1, 255], 'Invalid maxI value!!'    
        if (self.minI == -1 and self.maxI == 1):
            normalized_img = 2 * (self.image - np.min(self.image)) / (np.max(self.image) - np.min(self.image)) - 1
        elif (self.minI == 0 and self.maxI == 255):
            normalized_img = (self.image - np.min(self.image)) / (np.max(self.image) - np.min(self.image))
            normalized_img *= 255.0
        else:
            normalized_img = (self.image - np.min(self.image)) / (np.max(self.image) - np.min(self.image))
        return normalized_img

    def Focus_score(self):
        ''' Computes laplacian and returns the focus measure(ie variance for the image)
        args:
            image: image 
        Returns: 
            Focus measure of image(variance of laplacian)
        '''       
        Fc_score = (cv2.Laplacian(self.normalize_intensities(), cv2.CV_64F).var()
                )
        return Fc_score

    def local_Focus_score(self):
        image_data = self.normalize_intensities()
        assert self.scale > 1
        height, width =image_data.shape
        M = height // self.scale
        N = width // self.scale
        local_fcscore =  []
        for y in range(0,height,M):
            for x in range(0, width, N):
                image_tile = image_data[y:y+M,x:x+N]
                local_Fc_score = (cv2.Laplacian(image_tile, cv2.CV_64F).var())
                local_fcscore.append(local_Fc_score)
        return f'{float(np.median(local_fcscore)):.8f}', f'{float(np.mean(local_fcscore)):.8f}'

    def saturation_calculation(self):
        image_data = self.normalize_intensities()
        count_pixels  = np.product(self.image.shape)    
        if count_pixels == 0:
            maximum_percent = 0
            minimum_percent = 0
        else:
            count_maximumI_pixels = np.sum(image_data == np.max(image_data))
            count_minimumI_pixels = np.sum(image_data == np.min(image_data))
            maximum_percent = (
                100.0 * float(count_maximumI_pixels) / float(count_pixels)
            )
            minimum_percent = (
                100.0 * float(count_minimumI_pixels) / float(count_pixels)
            )
        return maximum_percent, minimum_percent

    def brisque_calculation(self):
        image_data = self.normalize_intensities()
        brisque_score  = brisque.score(image_data)
        return brisque_score

    def sharpness_calculation(self):
        ''' It is the non Reference based evaluation of the image quality). This method uses 
            the use of difference of differences in grayscale values of a median-filtered image as
             an indicator of edge sharpness
        args:
            image: image 
        Returns:
            Sharpness score of image
        '''
        dm = DOM()  
        sharpness_val = dm.get_sharpness(self.image)
        return sharpness_val

    def calculate_correlation_dissimilarity(self):   
        img = cv2.normalize(self.image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        xcor = set()       
        ycor = set()
        w, h = img.shape        
        patch = 10
        for i in range(0, 200):
            x1 = random.randint(i, w - patch -1)
            y1 = random.randint(i, h - patch  -1)
            x2, y2  = x1 + patch - 1, y1 + patch - 1
            xcor.add((x1, x2))
            ycor.add((y1, y2))

        patchedarea = []
        for x, y in zip(xcor, ycor):
            area = img[y[0]:y[1], x[0]:x[1]]
            patchedarea.append(area)            
               
        correlation, dissimilarity= [] , []
        for i, p in  enumerate(patchedarea):
            glcm = greycomatrix(patchedarea[i], distances=[5], angles=[0], levels=256,symmetric=True, normed=True)
            correlation.append(greycoprops(glcm, 'correlation')[0,0])
            dissimilarity.append(greycoprops(glcm, 'dissimilarity')[0, 0])       
        return np.mean(correlation), np.mean(dissimilarity)

    def rps(self):
        image_data = self.normalize_intensities()
        assert image_data.ndim == 2
        radii2 = (np.arange(image_data.shape[0]).reshape((image_data.shape[0], 1)) ** 2) + (
            np.arange(image_data.shape[1]) ** 2
        )
        radii2 = np.minimum(radii2, np.flipud(radii2))
        radii2 = np.minimum(radii2, np.fliplr(radii2))
        maxwidth = (
            min(image_data.shape[0], image_data.shape[1]) / 8.0
        )  # truncate early to avoid edge effects
        if image_data.ptp() > 0:
            image_data = image_data / np.median(abs(image_data - image_data.mean()))  # intensity invariant
        mag = abs(fft2(image_data - np.mean(image_data)))
        power = mag ** 2
        radii = np.floor(np.sqrt(radii2)).astype(np.int64) + 1
        labels = (
            np.arange(2, np.floor(maxwidth)).astype(np.int64).tolist()
        )  # skip DC component
        if len(labels) > 0:
            magsum = nd_sum(mag, radii, labels)
            powersum = nd_sum(power, radii, labels)
            return np.array(labels), np.array(magsum), np.array(powersum)
        return [2], [0], [0]

    def power_spectrum(self):
        image_data = self.normalize_intensities()
        radii, magnitude, power = self.rps()
        result = []        
        if sum(magnitude) > 0 and len(np.unique(image_data)) > 1:
            valid = magnitude > 0
            radii = radii[valid].reshape((-1, 1))
            power = power[valid].reshape((-1, 1))
            if radii.shape[0] > 1:
                idx = np.isfinite(np.log(power))
                powerslope = scipy.linalg.basic.lstsq(
                    np.hstack(
                        (
                            np.log(radii)[idx][:, np.newaxis],
                            np.ones(radii.shape)[idx][:, np.newaxis],
                        )
                    ),
                    np.log(power)[idx][:, np.newaxis],
                )[0][0]
            else:
                powerslope = 0
        else:
            powerslope = 0
        result += [float(powerslope)]
        return result[0]
       
def write_csv(*args):     
    with open(args[2], 'a+', newline='') as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        if os.path.getsize(args[2]) == 0:
            w.writerow(args[0])
        w.writerow(args[1])



 








    
        




