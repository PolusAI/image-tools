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
    parser.add_argument('--input', dest='input', type=str, help='input image collection', required=True)
    parser.add_argument('--notebook', dest='notebook', type=str, help='Jupyter notebook to run', required=True)
    parser.add_argument('--output', dest='output', type=str, help='output directory', required=True)
    args = parser.parse_args()
    
    input = args.input
    notebook = '/data/inputs/notebooks/' + args.notebook
    output = args.output

    print('Arguments:')    
    print(input)
    print(notebook)
    print(output)
    
    print('Beginning notebook execution...')
    process_start = time.time()

    with open(notebook) as nbfile:
        is_sos = json.load(nbfile)['metadata']['kernelspec']['language'] == 'sos'

    out = pm.execute_notebook(
       notebook,
       '/tmp/output.ipynb',
       engine_name="sos" if is_sos else None,
       parameters=dict(input_path=input, output_path=output)
    )

    print(out)
    
    print(' ')
    print('Execution completed in {} seconds!'.format(time.time() - process_start))
        
if __name__ == "__main__":
    main()