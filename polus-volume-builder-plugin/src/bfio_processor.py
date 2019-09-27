from bfio.bfio import BioReader
import numpy as np
import neuroglancer_scripts.accessor
import neuroglancer_scripts.precomputed_io
import neuroglancer_scripts.scripts.generate_scales_info
from pathlib import Path

# Conversion factors to nm
VOXEL_SCALES = {'m':  10**9,
                'cm': 10**7,
                'mm': 10**6,
                'µm': 10**3,
                'nm': 1,
                'Å':  10**-1}

def bfio_metadata_to_info(bfio_reader,
                          ignore_scaling=False,
                          input_min=None,
                          input_max=None,
                          options={}):
    
    voxel_sizes = [bfio_reader.num_x(),bfio_reader.num_y(),bfio_reader.num_z(),bfio_reader.num_c()]
    phys_x = bfio_reader.physical_size_x()
    phys_y = bfio_reader.physical_size_y()
    phys_z = bfio_reader.physical_size_z()
    voxel_dimensions = [voxel_sizes[0] * phys_x[0] * VOXEL_SCALES[phys_x[1]]]
    voxel_dimensions.append(voxel_sizes[1] * phys_y[0] * VOXEL_SCALES[phys_y[1]])
    if phys_z[0] == None or phys_z[0] == None:
        voxel_dimensions.append(voxel_sizes[2] * phys_y[0] * VOXEL_SCALES[phys_y[1]])
    else:
        voxel_dimensions.append(voxel_sizes[2] * phys_z[0] * VOXEL_SCALES[phys_z[1]])
    #voxel_resolutions = 
    
    formatted_info = """\
{{
    "type": "image",
    "num_channels": {num_channels},
    "data_type": "{data_type}",
    "scales": [
        {{
            "encoding": "raw",
            "size": {size},
            "resolution": {resolution},
            "voxel_offset": [0, 0, 0]
        }}
    ]
}}""".format(num_channels=bfio_reader.num_c(),
             data_type=bfio_reader._pix['type'],
             size=voxel_sizes,
             resolution=voxel_dimensions)
    return formatted_info

def bfio_to_precomputed(bfio_reader,
                        dest_url,
                        ignore_scaling=False,
                        input_min=None,
                        input_max=None,
                        load_full_volume=True,
                        options={}):
    #img = nibabel.load(volume_filename)
    accessor = neuroglancer_scripts.accessor.get_accessor_for_url(
        dest_url, options
    )
    info = bfio_metadata_to_info(bfio_reader)
    
    # Generate the full resolution json
    print(info)
    accessor.store_file("info_fullres.json",
                         info.encode("utf-8"),
                         mime_type="application/json")
    
    # Generate the info file with all scales
    neuroglancer_scripts.scripts.generate_scales_info.generate_scales_info(Path(dest_url).joinpath("info_fullres.json"),dest_url)
    
    try:
        precomputed_writer = neuroglancer_scripts.precomputed_io.get_IO_for_existing_dataset(
            accessor
        )
    except neuroglancer_scripts.accessor.DataAccessError as exc:
        print("No 'info' file was found (%s). You can generate one by "
                     "running this program with the --generate-info option, "
                     "then using generate_scales_info.py on the result",
                     exc)
        return 1
    except ValueError as exc:  # TODO use specific exception for invalid JSON
        print("Invalid 'info' file: %s", exc)
        return 1
    
    return None
    # return nibabel_image_to_precomputed(img, precomputed_writer,
    #                                     ignore_scaling, input_min, input_max,
    #                                     load_full_volume, options)

if __name__=="__main__":
    import javabridge as jutil
    import bioformats
    from pathlib import Path
    
    jutil.start_vm(class_path=bioformats.JARS)
    
    # Path to bioformats supported image
    image_path = Path('/mnt/3cc68e90-9b42-43df-b244-3492e361b382/WIPP/WIPP-plugins/collections/5d8955867d6eb900096f26c0/images')
    images = [i for i in image_path.iterdir() if "".join(i.suffixes)==".ome.tif"]
    
    #print(images)
    
    # Create the BioReader object
    bf = BioReader(str(images[0].absolute()))
    
    file_info = bfio_to_precomputed(bf,"./image1")
    
    # # Only load the first 256x256 pixels, will still load all Z,C,T dimensions
    # image = bf.read_image(X=(0,256),Y=(0,256))
    
    # # Only load the second channel
    # image = bf.read_image(C=[1])
    
    # # Done executing program, so kill the vm. If the program needs to be run
    # # again, a new interpreter will need to be spawned to start the vm.
    # javabridge.kill_vm()