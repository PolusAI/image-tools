import pytest
import javabridge as jutil
import bioformats
import bfio
from pathlib import Path
import matplotlib.pyplot as plt

""" Image path to test """
image_path = Path('/home/schaubnj/Desktop/Projects/polus-plugins/inputs/r001_c000_z000.ome.tif')

""" Fixtures """ 
# @pytest.fixture
def java_reader():
    return bfio.BioReader(image_path,backend='java')

# @pytest.fixture
def python_reader():
    return bfio.BioReader(image_path,backend='python')

p_reader = python_reader()

j_reader = java_reader()

plt.matshow(p_reader.read(X=[10240,11264],Y=[10240,11264]))
