import argparse
import os
import time
import pathlib
import papermill as pm
import json
import logging

def main():
    
    # intitialize logging
    logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)
    
    # Setup the Argument parsing
    parser = argparse.ArgumentParser(prog='script', description='Script to execute Jupyter Notebooks')

    # Parse input arguments from WIPP format: '--PARAMETER VALUE'
    parser.add_argument('--input-collection', dest='input_collection', type=str, help='input image collection', required=True)
    parser.add_argument('--input-notebook', dest='input_notebook', type=str, help='Jupyter notebook to run', required=True)
    parser.add_argument('--output-collection', dest='output_collection', type=str, help='output collection', required=True)
    parser.add_argument('--output-notebook', dest='output_notebook', type=str, help='executed notebook', required=True)
    parser.add_argument('--config-file', dest='config_file', type=str, help='configuration file', required=False)
    args = parser.parse_args()
    
    input_collection = args.input_collection
    input_notebook = os.path.join(args.input_notebook, 'notebook.ipynb')
    output_collection = args.output_collection
    output_notebook = os.path.join(args.output_notebook, 'notebook.ipynb')
    config_file=args.config_file

    
    logger.info('Arguments:')    
    logger.info('Input collection: {}'.format(input_collection))
    logger.info('Input notebook: {}'.format(input_notebook))
    logger.info('Config file: {}'.format(config_file))
    logger.info('Output collection: {}'.format(output_collection))
    logger.info('Output notebook: {}'.format(output_notebook))
    
    
    logger.info('Beginning notebook execution...')
    process_start = time.time()

    with open(input_notebook) as nbfile:
        is_sos = json.load(nbfile)['metadata']['kernelspec']['language'] == 'sos'

    if config_file == None:
        out = pm.execute_notebook(
        input_notebook,
        output_notebook,
        engine_name="sos" if is_sos else None,
        parameters=dict(input_path=input_collection, output_path=output_collection)
        )
    else:
        out = pm.execute_notebook(
        input_notebook,
        output_notebook,        
        engine_name="sos" if is_sos else None,
        parameters=dict(input_path=input_collection, output_path=output_collection, config_file_path=config_file)
        )        

    
    logger.info('Execution completed in {} seconds!'.format(time.time() - process_start))
        
if __name__ == "__main__":
    main()