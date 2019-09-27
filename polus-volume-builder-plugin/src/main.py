from neuroglancer_scripts import volume_reader
from neuroglancer_scripts.scripts.generate_scales_info import generate_scales_info
from bfio.bfio import BioReader

volume_reader.volume_file_to_info("/mnt/3cc68e90-9b42-43df-b244-3492e361b382/WIPP/WIPP-plugins/collections/5d8b7b736b88c300094892d9/metadata_files/avg152T1_LR_nifti2.nii","./avg152T1_LR_nifti2")
generate_scales_info("./avg152T1_LR_nifti2/info_fullres.json","./avg152T1_LR_nifti2")
volume_reader.volume_file_to_precomputed("/mnt/3cc68e90-9b42-43df-b244-3492e361b382/WIPP/WIPP-plugins/collections/5d8b7b736b88c300094892d9/metadata_files/avg152T1_LR_nifti2.nii","./avg152T1_LR_nifti2")