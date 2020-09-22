import numpy as np
from aicsimageio import AICSImage, omeTifWriter
from aicssegmentation.core.vessel import filament_2d_wrapper
from aicssegmentation.core.pre_processing_utils import intensity_normalization, image_smoothing_gaussian_3d, edge_preserving_smoothing_3d
from skimage.morphology import remove_small_objects 

def segment_image(image, config_data):  
    structure_channel = 0    
    struct_img0 = image[0,structure_channel,:,:,:].copy()
    intensity_scaling_param = config_data['intensity_scaling_param']
    if intensity_scaling_param[1] == 0:
        struct_img = intensity_normalization(struct_img0, scaling_param=intensity_scaling_param[:1])
    struct_img = intensity_normalization(struct_img0, scaling_param=intensity_scaling_param)
    gaussian_smoothing_sigma = config_data['gaussian_smoothing_sigma']
    if config_data['preprocessing_function'] == 'image_smoothing_gaussian_3d':
        structure_img_smooth = image_smoothing_gaussian_3d(struct_img, sigma=gaussian_smoothing_sigma)
    elif config_data['preprocessing_function'] == 'edge_preserving_smoothing_3d':
        structure_img_smooth = edge_preserving_smoothing_3d(struct_img)        
    #suggest_normalization_param(struct_img0)
    f2_param = config_data['f2_param']
    bw = filament_2d_wrapper(structure_img_smooth, f2_param)
    minArea = config_data['minArea']
    seg = remove_small_objects(bw>0, min_size=minArea, connectivity=1, in_place=False)
    seg = seg >0
    out_img=seg.astype(np.uint8)
    out_img[out_img>0]=255  
    return out_img  