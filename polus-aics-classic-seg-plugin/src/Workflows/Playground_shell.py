import numpy as np
import os
import cv2
from aicsimageio import AICSImage
from aicssegmentation.core.vessel import filament_2d_wrapper
from aicssegmentation.core.pre_processing_utils import intensity_normalization, image_smoothing_gaussian_3d
from aicssegmentation.core.utils import get_middle_frame, hole_filling, get_3dseed_from_mid_frame
from skimage.morphology import remove_small_objects, watershed, dilation, ball


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
        structure_img_smooth = image_smoothing_gaussian_3d(struct_img, sigma=gaussian_smoothing_sigma)

        middle_frame_method = config_data['middle_frame_method']
        mid_z = get_middle_frame(structure_img_smooth, method=middle_frame_method)

        f2_param = config_data['f2_param']
        bw_mid_z = filament_2d_wrapper(structure_img_smooth[mid_z,:,:], f2_param)

        hole_max = config_data['hole_max']
        hole_min = config_data['hole_min']

        bw_fill_mid_z = hole_filling(bw_mid_z, hole_min, hole_max)
        seed = get_3dseed_from_mid_frame(np.logical_xor(bw_fill_mid_z, bw_mid_z), struct_img.shape, mid_z, hole_min)
        bw_filled = watershed(struct_img, seed.astype(int), watershed_line=True)>0
        seg = np.logical_xor(bw_filled, dilation(bw_filled, selem=ball(1)))

        seg = seg > 0
        out=seg.astype(np.uint8)
        out[out>0]=255
        cv2.imwrite(os.path.join(outDir,f), out[0,:,:])