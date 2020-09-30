import os
import cv2
import numpy as np
from aicsimageio import AICSImage
from aicssegmentation.core.vessel import filament_2d_wrapper
from aicssegmentation.core.seg_dot import dot_2d_slice_by_slice_wrapper
from aicssegmentation.core.utils import hole_filling
from aicssegmentation.core.pre_processing_utils import intensity_normalization, image_smoothing_gaussian_slice_by_slice
from skimage.morphology import remove_small_objects, watershed, dilation, erosion, ball   



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
        structure_img_smooth = image_smoothing_gaussian_slice_by_slice(struct_img, sigma=gaussian_smoothing_sigma)
        
        s2_param = config_data['s2_param']
        bw_spot = dot_2d_slice_by_slice_wrapper(structure_img_smooth, s2_param)

        f2_param = config_data['f2_param']
        bw_filament = filament_2d_wrapper(structure_img_smooth, f2_param)

        bw = np.logical_or(bw_spot, bw_filament)

        fill_2d = config_data['fill_2d']
        if fill_2d == 'True':
            fill_2d = True
        elif fill_2d =='False':
            fill_2d = False
        fill_max_size = config_data['fill_max_size']
        minArea = config_data['minArea']

        bw_fill = hole_filling(bw, 0, fill_max_size, False)
        seg = remove_small_objects(bw_fill>0, min_size=minArea, connectivity=1, in_place=False)
    
        seg = seg >0
        out_img=seg.astype(np.uint8)
        out_img[out_img>0]=255 
        cv2.imwrite(os.path.join(outDir,f), out_img[0,:,:])