import time

def simple_read(reader):
    print()
    print('Using reading: {}'.format(reader))
    image_read = reader.read(X=[0,1024],Y=[0,1024])
    image_get = reader[0:1024,0:1024]
    
    print('image.shape: {}'.format(reader.shape))
    print('image.dtype: {}'.format(reader.dtype))
    print('image.read(X=[0,1024],Y=[0,1024]): ({},{})'.format(image_read.shape,image_read.dtype))
    print('image[0:1024,0:1024]: ({},{})'.format(image_get.shape,image_get.dtype))
    
def full_read(reader):
    print()
    
    start = time.time()
    for _ in range(10):
        image_read = reader.read()
    
    print('Finished loading the test image 10 times in {:.1f}ms.'.format((time.time() - start)*100))

class TestPythonReader():
    
    def test_read(self,python_reader):
        simple_read(python_reader)
        
    def test_time_read(self,python_reader):
        full_read(python_reader)
        
class TestJavaReader():
    
    def test_read(self,java_reader,jvm):
        simple_read(java_reader)
        
    def test_time_read(self,java_reader):
        full_read(java_reader)

if __name__=='__main__':
    import bfio
    import javabridge as jutil
    from pathlib import Path
    import matplotlib.pyplot as plt
    import numpy as np
    
    image_path = Path('../../input_image/r001_z000_y010_x010_c000.ome.tif')
    
    log_config = Path(__file__).parent.joinpath("log4j.properties")
    jutil.start_vm(args=["-Dlog4j.configuration=file:{}".format(bfio.LOG4J)],class_path=bfio.JARS)

    try:
        p_reader = bfio.BioReader(image_path,max_workers=1)

        j_reader = bfio.BioReader(image_path,backend='java',max_workers=1)

        fig,ax = plt.subplots(1,2)
        
        ax[0].imshow(p_reader[:].squeeze())
        ax[1].imshow(j_reader[:].squeeze())
        
        print(np.array_equal(p_reader[:].squeeze(),j_reader[:].squeeze()))
        
        plt.show()

    finally:
        jutil.kill_vm()
