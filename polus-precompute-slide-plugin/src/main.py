import logging, argparse, filepattern, bfio, utils, pathlib, multiprocessing
from preadator import ProcessManager

# Initialize the logger
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("main")
logger.setLevel(logging.INFO)

PyramidWriter = {
    'Neuroglancer': utils.NeuroglancerWriter,
    'DeepZoom': utils.DeepZoomWriter,
    'Zarr': utils.ZarrWriter
}

def main():
    
    # Set ProcessManager config and initialize
    ProcessManager.num_processes(multiprocessing.cpu_count())
    ProcessManager.num_threads(2*ProcessManager.num_processes())
    ProcessManager.threads_per_request(1)
    ProcessManager.init_processes('pyr')
    logger.info('max concurrent processes = %s', ProcessManager.num_processes())

    # Parse the input file directory
    fp = filepattern.FilePattern(input_dir,file_pattern)
    group_by = ''
    if 'z' in fp.variables and pyramid_type in ['Neuroglancer','Zarr']:
        group_by += 'z'
        logger.info('Stacking images by z-dimension for Neuroglancer/Zarr precomputed format.')
    elif 't' in fp.variables and pyramid_type == 'DeepZoom':
        group_by += 't'
        logger.info('Creating time slices by t-dimension for DeepZoom format.')
    else:
        logger.info(f'Creating one pyramid for each image in {pyramid_type} format.')
    
    depth = 0
    depth_max = 0
    image_dir = ''
    
    processes = []
    
    for files in fp(group_by=group_by):
        
        # Create the output name for Neuroglancer format
        if pyramid_type in ['Neuroglancer','Zarr']:
            try:
                image_dir = fp.output_name([file for file in files])
            except:
                pass
            
            if image_dir in ['','.*']:
                image_dir = files[0]['file'].name
                
            # Reset the depth
            depth = 0
            depth_max = 0
        
        pyramid_writer = None
        
        for file in files:
            
            with bfio.BioReader(file['file'],max_workers=1) as br:
                
                d_z = br.z
                
            depth_max += d_z
                
            for z in range(d_z):
                
                pyramid_args = {
                    'base_dir': output_dir.joinpath(image_dir),
                    'image_path': file['file'],
                    'image_depth': z,
                    'output_depth': depth,
                    'max_output_depth': depth_max
                }
                
                pw = PyramidWriter[pyramid_type](**pyramid_args)
                
                pw.write_slide()
                # ProcessManager.submit_process(pw.write_slide)
                
                depth += 1
                
                if pyramid_type == 'DeepZoom':
                    pw.write_info()
        
        # TODO: Aggregate labels for segmentation type
        if pyramid_type in ['Neuroglancer','Zarr']:
            pw.write_info()
    
    # ProcessManager.join_processes()

if __name__ == "__main__":
    
    # Setup the Argument parsing
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(
        prog='main',
        description='Generate a precomputed slice for Polus Volume Viewer.'
    )

    parser.add_argument('--inpDir', dest='input_dir', type=str,
                        help='Path to folder with CZI files', required=True)
    parser.add_argument('--outDir', dest='output_dir', type=str,
                        help='The output directory for ome.tif files', required=True)
    parser.add_argument('--pyramidType', dest='pyramid_type', type=str,
                        help='Build a DeepZoom or Neuroglancer pyramid', required=True)
    parser.add_argument('--filePattern', dest='file_pattern', type=str,
                        help='Filepattern of the images in input', required=False)
    parser.add_argument('--imageType', dest='image_type', type=str,
                        help='Either a image or a segmentation, defaults to image', required=False)

    '''Parse arguments'''
    args = parser.parse_args()
    input_dir = pathlib.Path(args.input_dir)
    output_dir = pathlib.Path(args.output_dir)
    
    # Validate the pyramid type
    pyramid_type = args.pyramid_type
    assert pyramid_type in ['Neuroglancer','DeepZoom','Zarr'], "pyramidType must be one of ['Neuroglancer','DeepZoom']"
    
    # Use a universal filepattern if none is provided
    file_pattern = args.file_pattern
    if file_pattern == None:
        file_pattern = '.*'
        
    # Default image_type to 'image'
    image_type = args.image_type
    if image_type == None:
        image_type = 'image'
    assert image_type in ['image','segmentation'], 'imageType must be one of ["image","segmentation"]'

    logger.info('input_dir = %s', input_dir)
    logger.info('output_dir = %s',output_dir)
    logger.info('pyramid_type = %s',pyramid_type)
    logger.info('image_type = %s', image_type)
    logger.info('file_pattern = %s', file_pattern)
    
    main()
