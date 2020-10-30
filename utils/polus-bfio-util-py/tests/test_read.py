import logging
logger = logging.getLogger('bfio')
logger.setLevel(logging.DEBUG)

import pytest, time, bioformats
import bfio
import javabridge as jutil
from pathlib import Path
import matplotlib.pyplot as plt

""" Image path to test """
image_path = Path('../../inputs/r001_c000_z000.ome.tif')

""" Fixtures """ 
@pytest.fixture
def java_reader():
    return bfio.BioReader(image_path,backend='java')

@pytest.fixture
def python_reader():
    return bfio.BioReader(image_path,backend='python')

def simple_read(reader):
    print()
    print('Using reading: {}'.format(reader))
    image_read = reader.read(X=[0,1024],Y=[0,1024])
    image_get = reader[0:1024,0:1024]
    
    print('image.shape: {}'.format(reader.shape))
    print('image.dtype: {}'.format(reader.dtype))
    print('image.read(X=[0,1024],Y=[0,1024]): ({},{})'.format(image_read.shape,image_read.dtype))
    print('image[0:1024,0:1024]: ({},{})'.format(image_get.shape,image_get.dtype))

class TestPythonReader():
    
    def test_read(self,python_reader):
        simple_read(python_reader)
        
class TestJavaReader():
    
    def test_read(self,java_reader,jvm):
        simple_read(java_reader)

# try:
#     p_reader = python_reader()

#     j_reader = java_reader()
    
    # j_reader = bfio.BioReader(str(image_path))

    # fig,ax = plt.subplots(1,2)
    
    # a = p_reader[0:1024,0:1024]
    
    # b = j_reader.read(X=[0,1024],Y=[0,1024])
    
    # p_reader[:] = a
    
    # start = time.time()
    # for _ in range(10):
    #     imagep = p_reader.read(X=[1000,1100],Y=[1000,1100]).squeeze()
    # print('Average loading time in Python: {:.2f} ms'.format((time.time() - start)*100))
    
    # start = time.time()
    # for _ in range(10):
    #     imagep = p_reader[1000:1100,1000:1100].squeeze()
    # print('Average loading time for __getitem__ in Python: {:.2f} ms'.format((time.time() - start)*100))
    
    # start = time.time()
    # for _ in range(10):
    #     imagej = j_reader.read(X=[1000,1100],Y=[1000,1100]).squeeze()
    # print('Average loading time in Java: {:.2f} ms'.format((time.time() - start)*100))
    
    # start = time.time()
    # for _ in range(10):
    #     imagej = j_reader[1000:1100,1000:1100].squeeze()
    # print('Average loading time for __getitem__ in Java: {:.2f} ms'.format((time.time() - start)*100))
    
    # ax[0].imshow(imagep)
    # ax[1].imshow(imagej)
    # plt.show()

# finally:
#     jutil.kill_vm()
