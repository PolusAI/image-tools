import numpy as np
import os
import cv2
from aicsimageio import AICSImage
from aicssegmentation.core.seg_dot import dot_3d_wrapper
from aicssegmentation.core.pre_processing_utils import intensity_normalization, image_smoothing_gaussian_3d
from skimage.morphology import remove_small_objects, binary_closing, ball , dilation   
from aicssegmentation.core.utils import topology_preserving_thinning
from aicssegmentation.core.MO_threshold import MO


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

        global_thresh_method = config_data['global_thresh_method'] 
        object_minArea = config_data['object_minArea'] 
        bw, object_for_debug = MO(structure_img_smooth, global_thresh_method=global_thresh_method, object_minArea=object_minArea, return_object=True)

        thin_dist_preserve = config_data['thin_dist_preserve']
        thin_dist = config_data['thin_dist']
        bw_thin = topology_preserving_thinning(bw>0, thin_dist_preserve, thin_dist)
        
        s3_param = config_data['s3_param']
        bw_extra = dot_3d_wrapper(structure_img_smooth, s3_param)

        bw_combine = np.logical_or(bw_extra>0, bw_thin)

        minArea = config_data['minArea']
        seg = remove_small_objects(bw_combine>0, min_size=minArea, connectivity=1, in_place=False)

        seg = seg > 0
        out=seg.astype(np.uint8)
        out[out>0]=255
        cv2.imwrite(os.path.join(outDir,f), out[0,:,:])      
        