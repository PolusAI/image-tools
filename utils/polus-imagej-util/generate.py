import os
import json
import shutil
import argparse
import logging
from pathlib import Path


"""This file uses the classes in populate.py and cookiecutter to automatically 
parse the ImageJ ops help and create plugins"""

if __name__ == '__main__':

    
    # Define the logger
    logger = logging.getLogger(__name__)
    
    # Set log level
    logger.setLevel(logging.DEBUG)
    
    # Define the logger format
    formatter = logging.Formatter(
        format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt = '%d-%b-%y %H:%M:%S'
        )
    
    # Define the logger file handler
    file_handler = logging.FileHandler('generate.log')
    
    # Set the filehandler format
    file_handler.setFormatter(formatter)
    
    # Add the file handler
    logger.addHandler(file_handler)
    
    # Define the parser
    parser = argparse.ArgumentParser(prog='main', description='Generate Plugins')

    # Add command-line argument for plugin name, docker repo and version
    parser.add_argument(
        '--plugins', 
        dest='plugins_to_generate', 
        type=str, 
        nargs="+", 
        default=["a", "b"],
        help='Plugins which will be generated', 
        required=True
    )

    # Parse the arguments
    args = parser.parse_args()
    plugins_to_generate = args.plugins_to_generate
    
    # Add plugins to generate to logger
    logger.debug('Plugins to Generate: {}'.format(plugins_to_generate))

    # Get the polus plugins directory
    polus_plugins_dir = Path(__file__).parents[2]

    # Add logger message
    logger.debug('Generating plugins with cookiecutter')

    # Get the generic.py file path so that it can find the cookietin directory
    base_path = Path(__file__).parent

    # Get path to cookietin dicrectory
    cookietin_path = base_path.joinpath('cookietin')

    # Get path to cookiecutter.json which is passed to cookiecutter
    cookiecutter_path = base_path.joinpath('cookiecutter.json')

    # Get list of all plugin directories in the cookietin directory 
    plugins = list(cookietin_path.iterdir())

    # Create path to imagej testing directory
    test_dir = Path(polus_plugins_dir.joinpath('imagej-testing'))

    # Check if the imagej testing directory already exists
    if test_dir.exists():
        # Remove directory if it exists
        shutil.rmtree(test_dir)

    # Create the imagej testing directory
    os.mkdir(test_dir)

    # Create path to the imagej shell script testing file
    shell_test_path = test_dir.joinpath('shell_test.py')

    # Create test summary file
    test_summary_path = shell_test_path.with_name('test-summary.log')

    # Create testing shell script file
    with open(shell_test_path, 'w') as fhand:
        fhand.write(
            'import os, sys\nfrom pathlib import Path\n' \
            'src_path = Path(__file__).parents[1]\nsys.path.append(str(src_path))\n'
        )
        fhand.close()

    # Creater counters for plugins and ops
    pluging_count = 0
    op_count = 0

    # Iterate over all plugin directories in the cookietin directory 
    for plugin in plugins:
        
        if plugin.name in plugins_to_generate:

            # Define the plugin dir path
            path = polus_plugins_dir.joinpath('polus-imagej-' + plugin.name.lower() + '-plugin')

            # If the plugin path is already a directory remove it recursively 
            if path.exists():
                shutil.rmtree(path)

            # Move the cookiecutter.json file for the current plug in to the 
            # polus-imagej-util directory overwriting last plugin json file
            shutil.copy(str(plugin.joinpath('cookiecutter.json')), cookiecutter_path)

            # Run the cookiecutter utility for the plugin
            os.system('cookiecutter {} --output-dir {} --no-input'.format(str(base_path), str(polus_plugins_dir)))

            # Use python black to format code
            os.system('black {}'.format(path))

            # Get the overloading methods from the op
            with open(cookiecutter_path, 'r') as f:
                op_methods = json.load(f)['_inputs']['opName']['options']
                f.close()

            # Open the shell script in append mode
            with open(shell_test_path, 'a') as fhand:

                # Get plugin dictionary key
                plugin_key = plugin.name.replace('-', '.')

                # Create a list of the operating sytem commands
                commands = ["python "+str(path)+"/tests/unit_test.py --opName '{}'".format(op) for op in op_methods]

                # Generate the shell script lines
                lines = ['os.system("{}")\n'.format(command) for command in commands]

                # Write command for each plugin to the shell script
                for line in lines:
                    fhand.write(line)
                    op_count += 1

                fhand.close()

                pluging_count += 1

    logger.debug('There were {} plugins generated\n'.format(pluging_count))
    logger.debug('There were {} plugin overloading methods created\n'.format(op_count))