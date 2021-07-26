#%%

import os
import imagej
import jpype
import scyjava
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from pathlib import Path
from bfio.bfio import BioReader, BioWriter

# Disable warnings when JVM starts
def disable_loci_logs():
    DebugTools = scyjava.jimport('loci.common.DebugTools')
    DebugTools.setRootLevel('WARN')
scyjava.when_jvm_starts(disable_loci_logs)


print('Starting JVM')
# Instantiate a pyimagej object
ij = imagej.init('sc.fiji:fiji:2.1.1+net.imagej:imagej-legacy:0.37.4', headless=True)

# Get path to input directory and macro
input_dir = Path(os.getcwd()).joinpath('input-images')
output_dir = input_dir.with_name('output-images')
macro_path = Path(__file__).with_name('smooth.ijm')

macro = """
#@ String input
run("Smooth");
saveAs("/home/ec2-user/polus-plugins/output-images/macro-test-smooth.png");
"""

# macro = """"""

# with open(macro_path) as f:
#     for line in f.readlines():
#         macro += line
#     f.close()
#     print('Running macro:\n')

args = {}

for inp in input_dir.iterdir():
    br = BioReader(inp)
    print('The {} shape is: {}'.format(inp, br.shape))
    img = np.squeeze(br[:,:,0:1,0,0])
    plt.figure()
    plt.imshow(img)
    plt.show()
    
    width = br.shape[0]
    height = br.shape[1]
    pixels = img
    #fp = ij.py.FloatProcessor(width, height, pixels, None)
    
    imagej_img = ij.py.to_dataset(img)
    
    img_plus = ij.py._numpy_to_dataset(img)
    
    #args['input'] = 'https://samples.fiji.sc/new-lenna.jpg'
    args['input'] = img_plus
    print('Processing image {}'.format(str(inp)))
    #args['output'] = str(output_dir.joinpath(inp))
    args['output'] = str(output_dir.joinpath('sample-img-smoothed.png'))
    ij.py.run_macro(macro, args)
    print(ij.py.active_dataset())
    


# with BioReader() as br:
#     pass

#img = io.imread(path_to_input)
#img = io.imread('https://samples.fiji.sc/new-lenna.jpg')
#img = np.mean(img[500:1000,300:850], axis=2)
#ij.py.show(img)


#print(ij.py.active_dataset())
#ij.py.to_java
#ij.py.run_macro(macro, args)

#ij.py.show(result)


print('Stopping JVM')
del ij
jpype.shutdownJVM()
    
# %%
