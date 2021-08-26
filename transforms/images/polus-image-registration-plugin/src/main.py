import argparse, logging, os
from pathlib import Path
from parser import parse_collection
import shutil
from preadator import ProcessManager
from image_registration import register

# Import environment variables
POLUS_LOG = getattr(logging,os.environ.get('POLUS_LOG','INFO'))
POLUS_EXT = os.environ.get('POLUS_EXT','.ome.tif')

# Initialize the logger
logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger("main")
logger.setLevel(POLUS_LOG)

def main(filePattern: str,
         inpDir: Path,
         registrationVariable: str,
         template: str,
         TransformationVariable: str,
         outDir: Path,
         method: str):
    
    # get template image path
    template_image_path=str(Path(inpDir).joinpath(template).absolute())
     
    # parse the input collection
    logger.info('Parsing the input collection and getting registration_dictionary')
    registration_dictionary=parse_collection(inpDir,filePattern,registrationVariable, TransformationVariable, template_image_path)
    
    logger.info('Iterating over registration_dictionary....')
    ProcessManager.init_processes(name='register')
    for registration_set,similar_transformation_set in registration_dictionary.items():
        
        # registration_dictionary consists of set of already registered images as well
        if registration_set[0]==registration_set[1]:            
            similar_transformation_set=similar_transformation_set.tolist()
            similar_transformation_set.append(registration_set[0])
            for image_path in similar_transformation_set:
                logger.info('Copying image {} to output directory'.format(image_path.name))
                shutil.copy2(image_path,str(Path(outDir).joinpath(image_path.name).absolute()))            
            continue
        
        # concatenate lists into a string to pass as an argument to argparse
        registration_string=' '.join([str(f) for f in registration_set])
        similar_transformation_string=' '.join([str(f) for f in similar_transformation_set])
        
        ProcessManager.submit_process(register,registration_string,similar_transformation_string,outDir,template,method)
        
    ProcessManager.join_processes()

if __name__=="__main__":

    # Setup the argument parsing
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='This plugin registers an image collection')
    parser.add_argument('--filePattern', dest='filePattern', type=str, help='Filename pattern used to separate data', required=True)
    parser.add_argument('--inpDir', dest='inpDir', type=str,help='Input image collection to be processed by this plugin', required=True)
    parser.add_argument('--registrationVariable', dest='registrationVariable', type=str, help='variable to help identify which images need to be registered to each other', required=True)
    parser.add_argument('--template', dest='template', type=str, help='Template image to be used for image registration', required=True)
    parser.add_argument('--TransformationVariable', dest='TransformationVariable', type=str,help='variable to help identify which images have similar transformation', required=True)
    parser.add_argument('--outDir', dest='outDir', type=str, help='Output collection', required=True)
    parser.add_argument('--method', dest='method', type=str, help='projective, affine, or partialaffine', required=True)
    
    # Parse the arguments     
    args = parser.parse_args()
    filePattern = args.filePattern
    logger.info('filePattern = {}'.format(filePattern))
    
    inpDir = args.inpDir
    # check if images folder is present in the input directory
    if (Path.is_dir(Path(inpDir).joinpath('images'))):
        inpDir= str(Path(inpDir).joinpath('images'))
        
    logger.info('inpDir = {}'.format(inpDir))
    registrationVariable = args.registrationVariable
    logger.info('registrationVariable = {}'.format(registrationVariable))
    template = args.template
    logger.info('template = {}'.format(template))
    TransformationVariable = args.TransformationVariable
    logger.info('TransformationVariable = {}'.format(TransformationVariable))
    outDir = args.outDir
    logger.info('outDir = {}'.format(outDir))     
    method = args.method
    logger.info('method = {}'.format(method)) 
    
    main(filePattern,
         inpDir,
         registrationVariable,
         template,
         TransformationVariable,
         outDir,
         method)
        