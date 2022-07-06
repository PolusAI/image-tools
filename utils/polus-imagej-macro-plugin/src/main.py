import os, logging, argparse
import imagej
import scyjava
import filepattern
import jpype
import numpy as np
from pathlib import Path
from bfio import BioReader, BioWriter


# Import environment variables
POLUS_LOG = getattr(logging, os.environ.get("POLUS_LOG", "INFO"))
POLUS_EXT = os.environ.get("POLUS_EXT", ".ome.tif")

# Initialize the logger
logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)

logger = logging.getLogger("main")
logger.setLevel(POLUS_LOG)

def close_image(image_title:str, ij: object) -> None:
    """
    Attempts to close an open image window via the macro command.
    """
    
    available_images = list(ij.WindowManager.getImageTitles())
    
    if image_title in available_images:
        
        logger.debug('Closing {}'.format(image_title))
        logger.debug('Current availalbe images:')
        logger.debug(available_images)
        
        # Attempt to close the image via macro command
        ij.py.run_macro(jpype.JString('close("{}");'.format(image_title)))
        
    else:
        logger.debug('Cannot close {} not in available images'.format(image_title))
    
# Define method to repeatedly run macro
def run_macro(
    numpy_input: np.array, 
    image_title: str,
    script: str, 
    ij: object,
    maxIterations: int
    ) -> np.array:
    """
    Runs macro on input image until the output received is modified version of
    the input or max iterations have been reached.
    """
    
    # Set input as copy of input for first while loop
    numpy_output = np.copy(numpy_input)
    
    # Set iteration counter
    i = 0
    
    # Run macro until input image does not equal output
    while np.array_equal(numpy_input, numpy_output):
        
        # Increment iteration counter
        i += 1
        logger.debug('Running macro attempt {}'.format(i))
        
        try:
            
            # Make sure the input image is closed before opening or re-opening
            close_image(image_title, ij)

            # HACK: Work around ImagePlus#show() failure if no ImagePlus objects 
            # are already registered.
            if ij.WindowManager.getIDList() is None:
                logger.debug('Creating dummy image...')
                ij.py.run_macro('newImage("dummy", "8-bit", 1, 1, 1);')
            

            logger.debug('Converting to ImagePlus Object...')
            # Convert to ImagePlus object
            java_input = ij.py.to_imageplus(numpy_input)
            
            logger.debug('Changing the title of the image')
            # Chanage the input image title
            java_input.setTitle(image_title)
            
            logger.debug('Registering as active image...')
            # HACK sets the ImagePlus object as the active image
            java_input.show()
            
            # Get the available images
            available_images = list(ij.WindowManager.getImageTitles())
            logger.debug('Available Images:')
            logger.debug(available_images)
            
            # Close all images except the input image
            for title in available_images:
                if title != image_title:
                    close_image(title, ij)
            
            # Check if the intended image is the active
            assert image_title == ij.py.active_imageplus(sync=False).getTitle()
            
            # Run the macro on the active image
            ij.py.run_macro(jpype.JString(script))
            
            logger.debug('Getting available images from WindowManager after macro...')
            for imp_id in ij.WindowManager.getIDList():
                logger.debug(ij.WindowManager.getImage(imp_id))
            
            logger.debug('Getting the output image...')
            
            # Get the active image after running the macro
            java_output = ij.WindowManager.getImage(jpype.JString(image_title + '-output'))
            
            logger.debug('Duplicating the output image')
            # HACK ensures the modifications to the image are sent to python
            java_duplicate = java_output.duplicate()
            java_duplicate.setTitle(jpype.JString(image_title + '-duplicate'))
            
            logger.debug('Sending image to python...')
            # Send the macro output to python
            xarr_output = ij.py.from_java(java_duplicate)

            logger.debug('Creating numpy output array...')
            # Convert the xarray to numpy array
            numpy_output = xarr_output.to_numpy()
        
        except:
            logger.debug('Macro attempt {} failed'.format(i))

        finally:
            # Use window manager to close all windows - clears up memory
            ij.WindowManager.closeAllWindows()
            
            # Terminate the plugin after max attempts to run macro on same image
            if i >= maxIterations:
                raise Exception('Failed to run macro on image {} after {} attempts'.format(image_title, i))

    return numpy_output
    
def main(inpDir, macro, outDir, maxIterations):
    
    # Load the macro script
    script = """"""

    with open(macro) as fhand:
        for line in fhand:
            script += line

    
    logger.debug('Macro script:\n' + script)

    # Infer the file pattern of the collection
    pattern_guess = filepattern.infer_pattern(inpDir.iterdir())
    
    # Instantiate the filepatter object
    fp = filepattern.FilePattern(inpDir, pattern_guess)

    # Get the collection's image paths
    image_paths = [f[0]['file'] for f in fp() if f[0]['file'].is_file()]
    
    # Disable the loci debug logs
    def disable_loci_logs():
        DebugTools = scyjava.jimport("loci.common.DebugTools")
        DebugTools.setRootLevel("WARN")
        
    scyjava.when_jvm_starts(disable_loci_logs)
    
    logger.debug('Is JVM running: {}'.format(jpype.isJVMStarted()))
    logger.info('Starting JVM...')
    
    # Instantiate imagej instance and launch JVM
    ij = imagej.init('sc.fiji:fiji:2.5.0', add_legacy=True)
    jpype.config.destroy_jvm = False
    
    # Define the ImageJ apps and versions
    apps = ['ImageJ1', 'ImageJ2', 'Fiji']
    versions = {app: None for app in apps}
    
    # Get the loaded apps and versions
    for app in apps:
        if ij.app().getApp(app) is not None:
            versions[app] = ij.app().getApp(app).getVersion()
        logger.debug('Loaded {} version {}'.format(app, versions[app]))
    
    # Iterate over the collection
    for path in image_paths:
        
        logger.info("Processing image: {}".format(path))
        
        # Load the current image
        with BioReader(path, backend='python') as br:
            numpy_input = np.squeeze(br[:, :, 0:1, 0, 0])
            metadata = br.metadata
            br.close()
        
        # Define the image stem
        path_stem = path.stem.split('.')[0]
        logger.debug('Path stem is {}'.format(path_stem))
        
        # Run the macro until correct output is returned form java
        numpy_output = run_macro(
            numpy_input=numpy_input,
            image_title=path_stem,
            script=script,
            ij=ij,
            maxIterations=maxIterations
        )
        
        # Make sure the input and output are not the same iamge
        assert not np.array_equal(numpy_input, numpy_output), 'The input and output images are identical'
        
        logger.info('Saving Image...')
        
        # Save the numpy output
        with BioWriter(outDir.joinpath(path.name), metadata=metadata) as bw:
            bw.Y = numpy_output.shape[0]
            bw.X = numpy_output.shape[1]
            bw.Z = 1
            bw.dtype = numpy_output.dtype
            bw[:] = numpy_output
            bw.close()
        
    logger.info('Complete!')


if __name__ == '__main__':
    
    # Setup Command Line Arguments
    logger.info("Parsing arguments...")
    
    # Instantiate argparser object
    parser = argparse.ArgumentParser(
        prog="main", description="ImageJ Macro Plugin"
    )

    # Add the plugin arguments
    parser.add_argument(
        "--inpDir", 
        dest="inpDir", 
        type=str, 
        help="Collection to be processed by this plugin", 
        required=True
        )
    
    parser.add_argument(
        "--macro", 
        dest="macro", 
        type=str, 
        help="The macro to apply to the collection", 
        required=True
    )
    
    parser.add_argument(
        "--outDir", 
        dest="outDir", 
        type=str, 
        help="Output collection", 
        required=True
    )

    parser.add_argument(
        "--maxIterations", 
        dest="maxIterations", 
        type=int, 
        help="The maximum number of macro attempts", 
        required=False
    )
    
    # Parse and log the arguments
    args = parser.parse_args()
    
    _inpDir = Path(args.inpDir)
    logger.info('inpDir = {}'.format(_inpDir))
    
    _macro = Path(args.macro)
    logger.info('macro = {}'.format(_macro))
        
    _outDir = Path(args.outDir)
    logger.info('outDir = {}'.format(_outDir))
    
    
    if args.maxIterations is not None:
        _maxIterations = args.maxIterations
    
    else:
        _maxIterations = 10
    logger.info('maxIterations = {}'.format(_maxIterations))
    
    
    main(
        inpDir=_inpDir, 
        macro=_macro, 
        outDir=_outDir, 
        maxIterations=_maxIterations
        )
