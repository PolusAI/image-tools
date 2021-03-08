from bfio.bfio import BioReader
import numpy as np
import json, copy, os
from pathlib import Path
import imageio
import filepattern
import os
import logging
import math
from concurrent.futures import ThreadPoolExecutor

# Initialize the logger    
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("utils")
logger.setLevel(logging.INFO)

def get_resolution(phys_y, phys_x, phys_z):
    """ 
    This function generates a resolution in nm 
    
    Parameters
    ----------
    phys_y : tuple
        Actual y dimension of input
    phys_x : tuple
        Actual x dimension of input
    phys_z : tuple
        Actual z dimension of input
    
    Returns
    -------
    resolution : list
        The integer values of resolution in nanometers in [Y, X, Z] order
        If Y and X resolutions are none, then default to 325 nm
        If Z resolution is none, then defaults to the average of Y and X
    """
    # Conversion factors to nm, these are based off of supported Bioformats length units
    UNITS = {'m':  10**9,
            'cm': 10**7,
            'mm': 10**6,
            'µm': 10**3,
            'nm': 1,
            'Å':  10**-1}

    if None in phys_y:
        phys_y = 325
    else:
        phys_y = phys_y[0] * UNITS[phys_y[1]]
    if None in phys_x:
        phys_x = 325
    else:
        phys_x = phys_x[0] * UNITS[phys_x[1]]
    if None in phys_z:
        phys_z = (phys_x + phys_y)/2
    else:
        phys_z = phys_z[0] * UNITS[phys_z[1]]
    
    return [phys_y, phys_x, phys_z]
    
