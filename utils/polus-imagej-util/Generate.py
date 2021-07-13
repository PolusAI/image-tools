import classes.Populate as cp
import os, jpype, imagej, shutil
from pathlib import Path
import time
    
if __name__ == '__main__':

    print('Starting JVM\n')
    
    # Start JVM
    ij = imagej.init('sc.fiji:fiji:2.1.1+net.imagej:imagej-legacy:0.37.4', headless=True)
    
    # Retreive all available operations from pyimagej
    #imagej_help_docs = scyjava.to_python(ij.op().help())
    #print(imagej_help_docs)
    
    print('Parsing imagej ops help\n')
    
    # Populate ops by parsing the imagej operations help
    populater = cp.Populate(ij)
    
    print('Building json templates\n')
    
    # Get the current working directory
    cwd = Path(os.getcwd())
    
    # Save a directory for the cookietin json files
    cookietin_path = cwd.joinpath('utils/polus-imagej-util/cookietin')
    
    # Build the json dictionary to be passed to the cookiecutter module 
    populater.buildJSON('Benjamin Houghton', 'benjamin.houghton@axleinfo.com', 'bthoughton', '0.1.1', cookietin_path)
    
    print('Shutting down JVM\n')
    
    # Remove the imagej instance
    del ij
    
    # Shut down JVM
    jpype.shutdownJVM()
    
    print('Generating plugins with cookiecutter\n')
    
    # Get the generic.py file path so that it can find the cookietin directory
    base_path = Path(__file__).parent
    
    # Get path to cookietin dicrectory
    cookietin_path = base_path.joinpath('cookietin')

    # Get path to cookiecutter.json file that lives in each of the 40 folders within cookietin
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

    # Create testing shell script file
    with open(shell_test_path, 'w') as fhand:
        fhand.write('import os, sys\nfrom pathlib import Path\nsrc_path = Path(__file__).parents[1]\nsys.path.append(str(src_path))\n')
        fhand.close()
    
    # Creater counters for plugins and ops
    pluging_count = 0
    op_count = 0
    
    # Iterate over all plugin directories in the cookietin directory 
    for plugin in plugins:
        
        # DELETE THIS LINE LATTER FOR UNIT TEST DEV
        plugins_to_generate = ['image-integral', 'image-distancetransform', 'filter-dog']
        plugins_to_generate = ['filter-dog', 'filter-addNoise', 'filter-convolve', 'filter-bilateral', 'filter-correlate']
        #if plugin.name == 'image-integral':
        if plugin.name in plugins_to_generate:
        #if True:
             
            # Create a path for the plugin
            path = Path(os.getcwd()).joinpath('polus-imagej-' + plugin.name.lower() + '-plugin')
            
            # If the plugin path is already a directory remove it and its children
            if path.exists():
                shutil.rmtree(path)
            
            # Move the cookiecutter.json file for the current plug in to the polus-imagej-util directory overwriting last plugin json file
            shutil.copy(str(plugin.joinpath('cookiecutter.json')), cookiecutter_path)
            
            # Run the cookiecutter utility for the plugin
            os.system('cookiecutter ./utils/polus-imagej-util/ --no-input')
            
            # Open the shell script in append mode
            with open(shell_test_path, 'a') as fhand:
                
                # Get plugin dictionary key
                plugin_key = plugin.name.replace('-', '.')
                
                # Get all availavle ops for the plugin
                ops = [op for op in populater._namespaces[plugin_key].supportedOps.keys()]
                
                # Create a list of the operating sytem commands
                commands = ['python '+str(path)+'/tests/unit_test2.py --opName '+op for op in ops]
                
                # Generate the shell script lines (each line is an os command to test a single op)
                lines = ["os.system('{}')\n".format(command) for command in commands]
                
                # Write command for each plugin to the shell script
                for line in lines:
                    fhand.write(line)
                    op_count += 1
                    
                fhand.close()
                
                pluging_count += 1

            # DELETE THIS LINE LATTER FOR UNIT TEST DEV
            #break
                
            
            
    print('There were {} plugins generated\n'.format(pluging_count))
    print('There were {} ops created\n'.format(op_count))
    
