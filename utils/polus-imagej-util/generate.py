import os
import json
import jpype
import shutil
import classes.populate as cp
from pathlib import Path


"""This file uses the classes in populate.py and cookiecutter to automatically 
parse the ImageJ ops help and create plugins"""

if __name__ == '__main__':

    # print('Starting JVM and parsing ops help\n')
    
    # # Populate ops by parsing the imagej operations help
    # populater = cp.Populate()
    
    # print('Building json templates\n')
    
    # Get the current working directory
    cwd = Path(os.getcwd())
    
    # # Save a directory for the cookietin json files
    # cookietin_path = cwd.joinpath('utils/polus-imagej-util/cookietin')
    
    # # Build the json dictionary to be passed to the cookiecutter module 
    # populater.build_json('Benjamin Houghton', 'benjamin.houghton@axleinfo.com', 'bthoughton', '0.2.0', cookietin_path)
    
    # print('Shutting down JVM\n')
    
    # # Remove the imagej instance
    # del populater._ij
    
    # # Shut down JVM
    # jpype.shutdownJVM()
    
    print('Generating plugins with cookiecutter\n')
    
    # Get the generic.py file path so that it can find the cookietin directory
    base_path = Path(__file__).parent
    
    # Get path to cookietin dicrectory
    cookietin_path = base_path.joinpath('cookietin')

    # Get path to cookiecutter.json file that lives in each of the 40 folders 
    # within cookietin
    cookiecutter_path = base_path.joinpath('cookiecutter.json')

    # Get list of all plugin directories in the cookietin directory 
    plugins = list(cookietin_path.iterdir()) 
    
    # Create path to imagej testing directory
    test_dir = Path(cwd.joinpath('imagej-testing'))
    
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
        
        plugins_to_generate = [
            'filter-dog',
            #'image-integral',
            #'filter-sobel'
        ]
    
        if plugin.name in plugins_to_generate:
        #if True:
             
            # Create a path for the plugin
            path = Path(os.getcwd()).joinpath('polus-imagej-' + plugin.name.lower() + '-plugin')
            
            # If the plugin path is already a directory remove it recursively 
            if path.exists():
                shutil.rmtree(path)
            
            # Move the cookiecutter.json file for the current plug in to the 
            # polus-imagej-util directory overwriting last plugin json file
            shutil.copy(str(plugin.joinpath('cookiecutter.json')), cookiecutter_path)
            
            # Run the cookiecutter utility for the plugin
            os.system('cookiecutter ./utils/polus-imagej-util/ --no-input')
            
            # Use python black to format code
            os.system('black {}'.format(path))
            
            print('\n')
            
            # Get the overloading methods from the op
            with open(cookiecutter_path, 'r') as f:
                op_methods = json.load(f)['_inputs']['opName']['options']
                f.close()
            
            # Open the shell script in append mode
            with open(shell_test_path, 'a') as fhand:
                
                # Get plugin dictionary key
                plugin_key = plugin.name.replace('-', '.')
                
                # Get all available ops for the plugin
                #ops = [op for op in populater._plugins[plugin_key].supported_ops.keys()]
                
                # Create a list of the operating sytem commands
                #commands = ["python "+str(path)+"/tests/unit_test.py --opName "+op for op in ops]
                commands = ["python "+str(path)+"/tests/unit_test.py --opName '{}'".format(op) for op in op_methods]
                
                # Generate the shell script lines (each line is an os command to test a single op)
                lines = ['os.system("{}")\n'.format(command) for command in commands]
                
                # Write command for each plugin to the shell script
                for line in lines:
                    fhand.write(line)
                    op_count += 1
                    
                fhand.close()
                
                pluging_count += 1
                
            
    print('There were {} plugins generated\n'.format(pluging_count))
    print('There were {} plugin overloading methods created\n'.format(op_count))