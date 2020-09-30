import numpy as np
import cv2
import os
from aicsimageio import AICSImage
from aicssegmentation.core.seg_dot import dot_3d, dot_3d_wrapper 
from aicssegmentation.core.pre_processing_utils import intensity_normalization, image_smoothing_gaussian_slice_by_slice, image_smoothing_gaussian_3d
from skimage.morphology import remove_small_objects, watershed, dilation, erosion, ball
from skimage.feature import peak_local_max
from skimage.measure import label
from scipy.ndimage import distance_transform_edt


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
        if intensity_scaling_param[1] == 0:
            struct_img = intensity_normalization(struct_img0, scaling_param=intensity_scaling_param[:1])
        else:
            struct_img = intensity_normalization(struct_img0, scaling_param=intensity_scaling_param)
            
        gaussian_smoothing_sigma = config_data['gaussian_smoothing_sigma'] 
        if config_data["gaussian_smoothing"] == "gaussian_slice_by_slice":
            structure_img_smooth = image_smoothing_gaussian_slice_by_slice(struct_img, sigma=gaussian_smoothing_sigma)
        else:
            structure_img_smooth = image_smoothing_gaussian_3d(struct_img, sigma=gaussian_smoothing_sigma)    
        s3_param = config_data['s3_param']
        bw = dot_3d_wrapper(structure_img_smooth, s3_param)
        minArea = config_data['minArea']
        Mask = remove_small_objects(bw>0, min_size=minArea, connectivity=1, in_place=False) 
        Seed = dilation(peak_local_max(struct_img,labels=label(Mask), min_distance=2, indices=False), selem=ball(1))
        Watershed_Map = -1*distance_transform_edt(bw)
        seg = watershed(Watershed_Map, label(Seed), mask=Mask, watershed_line=True)
        seg = remove_small_objects(seg>0, min_size=minArea, connectivity=1, in_place=False)
        seg = seg >0
        out=seg.astype(np.uint8)
        out[out>0]=255
        cv2.imwrite(os.path.join(outDir,f), out[0,:,:])
 
        