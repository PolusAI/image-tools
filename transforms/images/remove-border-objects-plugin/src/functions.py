import os
from bfio import BioReader, BioWriter
from pathlib import Path
import numpy as np
from skimage.segmentation import relabel_sequential


class Discard_borderobjects:
            
    """Discard objects which touches image borders and relabelling of objects.
    Args:
        inpDir (Path) : Path to label image directory
        outDir (Path) : Path to relabel image directory
        filename (str): Name of a label image
    Returns:
        label_image : ndarray of dtype int
        label_image, with discarded objects touching border
    """ 
    def __init__(self, inpDir, outDir, filename):
        self.inpDir = inpDir
        self.outDir=  outDir
        self.filename = filename
        self.imagepath = os.path.join(self.inpDir, self.filename)
        self.br_image = BioReader(self.imagepath)
        self.label_img = self.br_image.read().squeeze()

    def discard_borderobjects(self):
        """ This functions identifies which label pixels touches image borders and 
        setting the values of those label pixels to background pixels values which is 0
        """
        borderobj = list(self.label_img[0, :])
        borderobj.extend(self.label_img[:, 0])
        borderobj.extend(self.label_img[- 1, :])
        borderobj.extend(self.label_img[:, - 1])
        borderobj = np.unique(borderobj).tolist()

        for obj in borderobj:
            self.label_img[self.label_img == obj] = 0

        return self.label_img

    def relabel_sequential(self):
        """ Sequential relabelling of objects in a label image
        """
        relabel_img, _, inverse_map  = relabel_sequential(self.label_img)
        return relabel_img, inverse_map


    def save_relabel_image(self, x):
        """ Writing images with relabelled and cleared border touching objects
        """
        with BioWriter(file_path = Path(self.outDir, self.filename),
                    backend='python',
                    metadata  = self.br_image.metadata, 
                    X=self.label_img.shape[0],  
                    Y=self.label_img.shape[0], 
                    dtype=self.label_img.dtype) as bw:
            bw[:] = x
            bw.close() 
        return       