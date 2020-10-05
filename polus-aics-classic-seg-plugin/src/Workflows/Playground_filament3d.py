import numpy as np
import os
import cv2
from aicsimageio import AICSImage
from aicssegmentation.core.vessel import filament_3d_wrapper
from aicssegmentation.core.pre_processing_utils import intensity_normalization, image_smoothing_gaussian_3d, edge_preserving_smoothing_3d
from skimage.morphology import remove_small_objects 


def segment_images(inpDir, outDir, config_data): 

    inpDir_files = os.listdir(inpDir)
    for i,f in enumerate(inpDir_files):
        # Load an image
        #br = BioReader(Path(inpDir).joinpath(f))
        #image = np.squeeze(br.read_image())
        reader = AICSImage(os.path.join(inpDir,f)) 
        image = reader.data.astype(np.float32)

        structure_channel = 0    
        struct_img0 = image[0,structure_channel,:,:,:].copy()
        intensity_scaling_param = config_data['intensity_scaling_param']
        struct_img = intensity_normalization(struct_img0, scaling_param=intensity_scaling_param)
        gaussian_smoothing_sigma = config_data['gaussian_smoothing_sigma']
 
        if config_data['preprocessing_function'] == 'image_smoothing_gaussian_3d':
            structure_img_smooth = image_smoothing_gaussian_3d(struct_img, sigma=gaussian_smoothing_sigma)
        elif config_data['preprocessing_function'] == 'edge_preserving_smoothing_3d':
            structure_img_smooth = edge_preserving_smoothing_3d(struct_img)        

        
        f3_param = config_data['f3_param']
        
        bw = filament_3d_wrapper(structure_img_smooth, f3_param)
  
        minArea = config_data['minArea']
 
        seg = remove_small_objects(bw>0, min_size=minArea, connectivity=1, in_place=False)
        seg = seg >0
        out_img=seg.astype(np.uint8)
        out_img[out_img>0]=255  
        cv2.imwrite(os.path.join(outDir,f), out_img[0,:,:])