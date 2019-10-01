import argparse
import os
import time
import pathlib
import papermill as pm
import json

def main():
    # Setup the Argument parsing
    parser = argparse.ArgumentParser(prog='script', description='Script to execute Jupyter Notebooks')

    # Parse input arguments from WIPP format: '--PARAMETER VALUE'
    parser.add_argument('--input-collection', dest='input_collection', type=str, help='input image collection', required=True)
    parser.add_argument('--input-notebook', dest='input_notebook', type=str, help='Jupyter notebook to run', required=True)
    parser.add_argument('--output-collection', dest='output_collection', type=str, help='output collection', required=True)
    parser.add_argument('--output-notebook', dest='output_notebook', type=str, help='executed notebook', required=True)
    args = parser.parse_args()
    
    input_collection = args.input_collection
    input_notebook = os.path.join(args.input_notebook, 'notebook.ipynb')
    output_collection = args.output_collection
    output_notebook = os.path.join(args.output_notebook, 'notebook.ipynb')

    print('Arguments:')    
    print(f'Input collection: {input_collection}')
    print(f'Input notebook: {input_notebook}')
    print(f'Output collection: {output_collection}')
    print(f'Output notebook: {output_notebook}')
    
    print('Beginning notebook execution...')
    process_start = time.time()

    with open(input_notebook) as nbfile:
        is_sos = json.load(nbfile)['metadata']['kernelspec']['language'] == 'sos'

    out = pm.execute_notebook(
       input_notebook,
       output_notebook,
       engine_name="sos" if is_sos else None,
       parameters=dict(input_path=input_collection, output_path=output_collection)
    )

    print(out)
    
    print(' ')
    print('Execution completed in {} seconds!'.format(time.time() - process_start))
        
if __name__ == "__main__":
    main()