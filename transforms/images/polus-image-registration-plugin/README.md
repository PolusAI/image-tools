# Polus Projective Transformation Image Registration Plugin

WIPP Plugin Title : Image Registration Plugin 

Contact [Gauhar Bains](mailto:gauhar.bains@labshare.org) for more information.

For more information on WIPP, visit the [official WIPP page](https://isg.nist.gov/deepzoomweb/software/wipp).

## Description

This plugin registers an image collection. First it parses the image collection using `parser.py` into registration sets. Each registration set consists of: moving image, template image, similiar transformation images. The registration algorithm(explained in the next section, can be found in `image_registration.py`) registers the moving image with template image and stores the transformation required to do so. This stored transformation is used to transform each image in similar transformation list. 

## Algorithm

### To do
1. Find a better way to handle poorly correlated transforms. 
2. Find a more scalable approach to do rough transformation. The algorithm has been tested on images of size around 1.2 Gigapixel but a better approach may be needed for images significantly larger than these.

### Parsing 
The parsing algorithm uses the functions from the `file_pattern utility`. It takes the following inputs : Filename pattern, registration variable, similar transformation variable. The registration variable helps determine the moving and the template images where as the similar transformation variable helps determine the similar transformation images. Note: The code produces the expected output when len(registration_variable)==len(similarity_variable)==1. The code will NOT spit out an error when the more than one variable is passed as registration or similarity variable, but additional testing needs to be done for this usecase.  

Some sample text files can be found in the examples folder. Short example shown below:    

Parsing example :   
  
`Inputs:`  
Filepattern :   `x{xxx}_y{yyy}_z{zzz}_c{ccc}_t{ttt}.ome.tif`  
Registration_variable :  `t`   
similar_transformation_variable : `c`  
template_ :  `x001_y001_z001_c001_t001.ome.tif`    

`Output set 1 :`   
Template Image:  x001_y001_z001_c001_t001.ome.tif  
Moving Image:  x001_y001_z001_c001_t002.ome.tif  
Similar Transformation Images :   [ x001_y001_z001_c002_t002.ome.tif , x001_y001_z001_c003_t002.ome.tif ]  

`Output set 2:`    
Template Image:  x001_y002_z001_c001_t001.ome.tif    
Moving Image:   x001_y002_z001_c001_t002.ome.tif    
Similar Transformation Images :  [ x001_y002_z001_c002_t002.ome.tif , x001_y002_z001_c003_t002.ome.tif ]      



### Registration 
The registration algorithm is present in `image_registration.py`. It uses projective transformation(Homography matrix) to alter the moving image and align it with the reference image. Background information about homography can be found here: https://en.wikipedia.org/wiki/Homography .    
The moving image undergoes 2 transformations:     
1. `Rough Transformation` : In this the whole moving image is transformed using the homography matrix calculated between the entire moving and template image.
2. `Fine Transformation` : To carry out fine transformation ,the homography matrix is found between the corresponding tiles of the roughly transformed moving image and the template image. Each image is divided into 4 tiles. 

To find the homography matrix(for fine or rough tranformation), we need coordinates of atleast 4 matching points in the template and the moving image. To do this the ORB feature detector has been used. However, its computationally very expensive to run feature matching on large images(our test data consists of 1.3 gigapixel images). To overcome this, the homography matrix at every step of our algorithm has been calculated between scaled down versions( 16 * 16 times smaller) of the respective images. To use this homography matrix on actual sized images, the matrix is scaled up using a scale matrix.  The proof for upscaling a homography matrix is shown below.   

`Proof` :  

Credit for proof : https://stackoverflow.com/questions/21019338/how-to-change-the-homography-with-the-scale-of-the-image/56623249    
  
![homography](https://user-images.githubusercontent.com/48079888/78402511-b04d8200-75c8-11ea-9d22-cee13f3912db.gif)  
   







## Building

To build the Docker image for the conversion plugin, run
`./build-docker.sh`.

## Install WIPP Plugin

If WIPP is running, navigate to the plugins page and add a new plugin. Paste the contents of `plugin.json` into the pop-up window and submit.

## Options

This plugin takes one input argument and one output argument:

| Name          | Description             | I/O    | Type   |
|---------------|-------------------------|--------|--------|
| `--filePattern` | Filename pattern used to separate data | Input | string |
| `--inpDir` | Input image collection to be processed by this plugin | Input | collection |
| `--registrationVariable` | variable to help identify which images need to be registered to each other | Input | string |
| `--template` | Template image to be used for image registration | Input | string |
| `--TransformationVariable` | variable to help identify which images have similar transformation | Input | string |
| `--outDir` | Output collection | Output | collection |

