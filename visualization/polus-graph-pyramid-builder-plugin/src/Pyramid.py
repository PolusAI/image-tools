import os
import numpy as np

from PIL import Image

from concurrent.futures import ThreadPoolExecutor

import logging
POLUS_LOG = getattr(logging,os.environ.get('POLUS_LOG', 'INFO'))
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("pyramid")
logger.setLevel(POLUS_LOG)

def _avg2(image):
    """ Average pixels with optical field of 2x2 and stride 2 """
    
    # Convert 32-bit pixels to prevent overflow during averaging
    image = image.astype(np.uint32) # averaging four numbers, each image is uint8 -> 8*4=32

    imageshape0 = image.shape[0]
    imageshape1 = image.shape[1]

    # Get the height and width of each image to the nearest even number
    y_max = imageshape0 - imageshape0 % 2
    x_max = imageshape1 - imageshape1 % 2
    
    # Perform averaging
    avg_img = np.zeros(np.ceil([image.shape[0]/2,image.shape[1]/2,*image.shape[2:]]).astype(np.uint32))
    avg_img[0:int(y_max/2),0:int(x_max/2),...]= (image[0:y_max-1:2,0:x_max-1:2,...] + \
                                                 image[1:y_max:2,0:x_max-1:2,...] + \
                                                 image[0:y_max-1:2,1:x_max:2,...] + \
                                                 image[1:y_max:2,1:x_max:2,...]) / 4
    
    # The next if statements handle edge cases if the height or width of the
    # image has an odd number of pixels
    if y_max != imageshape0:
        avg_img[-1,:int(x_max/2),...] = (image[-1,0:x_max-1:2,...] + \
                                         image[-1,1:x_max:2,...]) / 2
    if x_max != imageshape1:
        avg_img[:int(y_max/2),-1,...] = (image[0:y_max-1:2,-1,...] + \
                                         image[1:y_max:2,-1,...]) / 2
    if y_max != imageshape0 and x_max != imageshape1:
        avg_img[-1,-1,...] = image[-1,-1,...]

    return avg_img.astype(np.uint8)

class GraphPyramid():

    """ This class generates the data structure needed to run DeepZoom on the Plots Created"""

    def __init__(self, output_dir  : str, 
                       output_name : str, 
                       ngraphs     : int, 
                       axisnames   : list, 
                       CHUNK_SIZE  : int):
        """
        Inputs:
            output_dir - location of where all the data for the pyramid are stored
            output_name - name of base dir to save the scaled directories
            ngraphs     - the total number of graphs that should be generated
            axisnames   - name of images in scaled directories
            CHUNK_SIZE  - size of images
        """

        self.CHUNK_SIZE = CHUNK_SIZE
        self.HALF_CHUNK_SIZE = CHUNK_SIZE//2

        self.ngraphs    = ngraphs
        self.fig_dim    = [int(np.ceil(np.sqrt(ngraphs))), int(np.round(np.sqrt(ngraphs)))] #number of rows and columns in DeepZoom
        self.sizes      = [dim*self.CHUNK_SIZE for dim in self.fig_dim]
        self.num_scales = np.floor(np.log2(max(self.sizes))).astype('int')+1
        self.image_extension = 'png'
        self.base_imagenames = {}

        with open(os.path.join(output_dir, f"{output_name}.dzi"), "w") as out_dzi:
            out_dzi.write(f'<?xml version="1.0" encoding="utf-8"?><Image TileSize="{self.CHUNK_SIZE}' + \
                            f'" Overlap="0" Format="{self.image_extension}" xmlns="http://schemas.microsoft.com/deepzoom/2008">' + \
                            f'<Size Width="{self.sizes[0]}" Height="{self.sizes[1]}"/></Image>')

        with open(os.path.join(output_dir, f"{output_name}.csv"), "w") as out_csv:
            out_csv.write('dataset_id, x_axis_id, y_axis_id, x_axis_name, y_axis_name, ' + \
                            'title, length, width, global_row, global_col\n')
            
            for fig_row in range(self.fig_dim[0]):
                for fig_col in range(self.fig_dim[1]):
                    combo = axisnames[fig_row*self.fig_dim[0]+fig_col]
                    self.base_imagenames[combo] = f'{fig_row}_{fig_col}.{self.image_extension}'
                    out_csv.write(f'1, {fig_col}, {fig_row}, {combo[0]}, {combo[1]}, ' + \
                                    f'default title, {self.CHUNK_SIZE}, {self.CHUNK_SIZE}, ' + \
                                    f'{fig_col}, {fig_row}\n')

        self.figs_path = os.path.join(output_dir, f"{output_name}_files")
        if not os.path.exists(self.figs_path):
            os.mkdir(self.figs_path)

        self.bottom_pyramidDir = os.path.join(self.figs_path, str(self.num_scales))
        if not os.path.exists(self.bottom_pyramidDir):
            os.mkdir(self.bottom_pyramidDir)

    def build_nextlayer(self, x_node       : int, 
                              y_node       : int, 
                              scale_dir    : str, 
                              readfrom_dir : str):
        """ This function creates one image in the next layer of the pyramid by averaging the 
        four relevant images in z order. 
        
        Inputs:
            x_node       - column node of image in layer
            y_node       - row node of image in layer
            scale_dir    - directory in which the averaged image is saved in
            readfrom_dir - directory of the four images being read from
        """

        try:
            top_left     = os.path.join(readfrom_dir, f"{x_node}_{y_node}.{self.image_extension}")
            top_right    = os.path.join(readfrom_dir, f"{x_node+1}_{y_node}.{self.image_extension}")
            bottom_left  = os.path.join(readfrom_dir, f"{x_node}_{y_node+1}.{self.image_extension}")
            bottom_right = os.path.join(readfrom_dir, f"{x_node+1}_{y_node+1}.{self.image_extension}")

            output_node = f"{x_node//2}_{y_node//2}.{self.image_extension}"

            rightexists = False
            bottomexists = False

            # Need to check if the neighbor images exist for z map ordering
                # https://en.wikipedia.org/wiki/Z-order
            if os.path.exists(top_right):
                rightexists = True
                topright_array = _avg2(np.array(Image.open(top_right)))
            if os.path.exists(bottom_left):
                bottomexists = True
                bottomleft_array = _avg2(np.array(Image.open(bottom_left)))

            # The top left must exist.  Otherwise the entire New Node is empty
            if not os.path.exists(top_left):
                logger.debug(f"{top_left} does not exist")
                return
            
            # logger.debug(f"right exists - {rightexists}, bottom exists - {bottomexists}")
            topleft_array = _avg2(np.array(Image.open(top_left)))

            logger.debug(f"SCALE - {os.path.basename(scale_dir)}: \n" + 
                f"\tTOP LEFT NODE: {x_node}_{y_node} - Should Always Exist\n" +
                f"\tTOP RIGHT NODE: {x_node}_{y_node+1} - Does Exist? {rightexists}\n" +
                f"\tBOTTOM LEFT NODE: {x_node+1}_{y_node} - Does Exist? {bottomexists}\n" +
                f"\tBOTTOM RIGHT NODE: {x_node+1}_{y_node+1} - Not needed for Initialization\n" + 
                f"Create Node {x_node//2}_{y_node//2} from {readfrom_dir}")

            # trying to append the images from the pyramid below in Z order
                # z map ordering for four images:
                    # 1) top left - 0,0
                    # 2) top right - 0,1
                    # 3) bottom left - 1,0
                    # 4) bottom right - 1,1
            if (rightexists and bottomexists):

                output_image = np.zeros((self.HALF_CHUNK_SIZE+bottomleft_array.shape[0], 
                                         self.HALF_CHUNK_SIZE+topright_array.shape[1], 
                                         *topleft_array.shape[2:]), dtype=np.uint8)
                logger.debug(f"Output Shape: {output_image.shape}")

                output_image[0:self.HALF_CHUNK_SIZE, 0:self.HALF_CHUNK_SIZE, ...] = topleft_array

                logger.debug(f"top right array shape: {topright_array.shape}, " + \
                                f"add in: (0:{self.HALF_CHUNK_SIZE}, {self.HALF_CHUNK_SIZE}:{self.HALF_CHUNK_SIZE+topright_array.shape[1]})")
                logger.debug(f"bottom left array shape: {bottomleft_array.shape}, " + \
                                f"add in: ({self.HALF_CHUNK_SIZE}:{self.HALF_CHUNK_SIZE+bottomleft_array.shape[0]}, 0:{self.HALF_CHUNK_SIZE})")

                output_image[0:self.HALF_CHUNK_SIZE, self.HALF_CHUNK_SIZE:(self.HALF_CHUNK_SIZE+topright_array.shape[1]), ...] = topright_array
                output_image[self.HALF_CHUNK_SIZE:(self.HALF_CHUNK_SIZE+bottomleft_array.shape[0]), 0:self.HALF_CHUNK_SIZE, ...] = bottomleft_array

                # if the top right and bottom left do not exist, then the bottom right should not exist -- this follows z order too
                if os.path.exists(bottom_right):
                    logger.debug(f"ALL 4/4 NODES USED TO CREATE (scale {os.path.basename(scale_dir)}): {output_node}")
                    output_image[self.HALF_CHUNK_SIZE:output_image.shape[0], self.HALF_CHUNK_SIZE:output_image.shape[1], ...] = \
                        _avg2(np.array(Image.open(bottom_right)))

                output_im = Image.fromarray(output_image)

            # the relevant dimensions should be the same, so we can append without initialization
            elif (rightexists and not bottomexists):
                logger.debug("appending (TOP LEFT & TOP RIGHT) nodes")
                output_im = Image.fromarray(np.append(topleft_array, topright_array, axis=1))
            elif (not rightexists and bottomexists):
                logger.debug("appending (TOP LEFT & BOTTOM LEFT) nodes")
                output_im = Image.fromarray(np.append(topleft_array, bottomleft_array, axis=0))
            else:
                logger.debug("no appending nodes")
                output_im = Image.fromarray(topleft_array)

            output_im.save(os.path.join(scale_dir, output_node))
            logger.info(f"SAVED: {os.path.join(scale_dir, output_node)}")

        except Exception as e:
            raise ValueError(f"something went wrong - {e}")


    def build_thepyramid(self):
        """ This function the remaining pyramid once the bottom layer is populated"""

        assert len(os.listdir(self.bottom_pyramidDir)) == self.ngraphs, \
            f"Program generated {len(os.listdir(self.bottom_pyramidDir))} plots, but expected {self.ngraphs} plots"

        image_nodes = self.fig_dim

        # the data structure used to generate pyramids: 
            # https://en.wikipedia.org/wiki/Quadtree
        
        readfrom_dir = self.bottom_pyramidDir
        for scale in reversed(range(self.num_scales)): # bottom up traversal through the QuadTree
            logger.info(f"Building Scale: {scale} ...")
            scale_dir = os.path.join(self.figs_path, str(scale))
            logger.debug(scale_dir)
            if not os.path.exists(scale_dir):
                logger.debug(f"Making {scale_dir}")
                os.mkdir(scale_dir)

            with ThreadPoolExecutor(max_workers=os.cpu_count()-1) as executor:
                for x in range(0, image_nodes[0]+1, 2):
                    for y in range(0, image_nodes[1]+1, 2):
                        executor.submit(self.build_nextlayer, x_node=x, y_node=y, scale_dir=scale_dir, readfrom_dir=readfrom_dir)

            logger.debug(f"Done Building Scale: {scale} ...\n\n")
            readfrom_dir = scale_dir
            # for every layer generated, we output four times as less images,
               # because we are usually combining four images into one image by averaging them all together
            image_nodes = [max(node//2,1) for node in image_nodes] # there should always be at least one node in every dimension