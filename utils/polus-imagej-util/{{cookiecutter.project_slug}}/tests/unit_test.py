import unittest, json, tempfile, numpy, os, shutil, logging, sys, time, jpype, imagej, argparse
from pathlib import Path
from bfio.bfio import BioReader, BioWriter

# Get pluging directory
plugin_dir = Path(__file__).parents[1]
sys.path.append(str(plugin_dir))

from src.main import main

class UnitTest(unittest.TestCase):
    
    # Create a random image to be used for plugin testing
    infile = None
    outfile = None
    image_size = 2048
    image_shape = (image_size, image_size)
    random_image = numpy.random.randint(
        low = 0,
        high = 255,
        size = image_shape,
        dtype = numpy.uint8
    )
    
    @classmethod
    def setUpClass(cls) -> None:
        
        # Create input and output path objects for the randomly generated image file
        cls.inputPath = Path(__file__).parent.joinpath('input/random.ome.tif')
        cls.outputPath = Path(__file__).parent.joinpath('output/random.ome.tif')
        
        # Check if "input" is a sub-directory of "tests"
        if cls.inputPath.parent.exists():
            
            # Remove the "input" sub-directory
            shutil.rmtree(cls.inputPath.parent)
            
        # Check if "output" is a sub-directory of "tests"
            if cls.outputPath.parent.exists():
            
                # Remove the "input" sub-directory
                shutil.rmtree(cls.outputPath.parent)
        
        # Create input and output sub-directories in tests
        os.mkdir(cls.inputPath.parent)
        os.mkdir(cls.outputPath.parent)
        
        # Create a BioWriter object to write the ramdomly generated image file to tests/input dir
        with BioWriter(cls.inputPath) as writer:
            writer.X = cls.image_shape[0]
            writer.Y = cls.image_shape[1]
            writer[:] = cls.random_image[:]
            writer.close()
        
        # Set up logger for failed tests
        cls.logger = logging.getLogger(__name__)
        cls.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(levelname)s:%(message)s')
        file_handler = logging.FileHandler('test.log')
        file_handler.setFormatter(formatter)
        cls.logger.addHandler(file_handler)
        
        return
    
    @classmethod
    def tearDownClass(cls) -> None:
        return
    
    def test_plugin(self):
        
        # Get path to the plugin json template file
        plugin_json_path = Path(__file__).parents[1].joinpath('plugin.json')
        
        # Open json file and get names of all ops in namespace
        with open(plugin_json_path, 'r') as fhand:
            plugin_json = json.load(fhand)
            ops = plugin_json['inputs'][0]['options']['values']
            namespace = plugin_json['name']
        
        
        sample_input = Path(__file__).parents[2].joinpath('input-images')
        sample_output = sample_input.with_name('output-images')
        
        # Run op with randomly generated image file
        try:
            # Run the op and log successful output
            main(op, self.inputPath.parent, self.outputPath.parent)
            self.logger.info('The {} with op option {} was successful'.format(namespace, op))
            
        except:
            self.logger.info('FAILURE: The {} with op option {} was not successful'.format(namespace, op))
            self.logger.info(sys.exc_info())
            
if __name__ == '__main__':
    
    # Instantiate a parser for command line arguments
    parser = argparse.ArgumentParser(prog='unit_test', description='DefaultIntegralImg, WrappedIntegralImg')
    
    # Add command-line argument for each of the input arguments
    parser.add_argument('--opName', dest='opName', type=str,
                        help='Operation to test', required=True)
    

    """ Parse the arguments """
    args = parser.parse_args()
    
    # Input Args
    op = args.opName
    
    del sys.argv[1:]
    unittest.main()