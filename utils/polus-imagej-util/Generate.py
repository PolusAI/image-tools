import classes.Populate as cp
import os, jpype, imagej, shutil
from pathlib import Path
    
if __name__ == '__main__':

    print('Starting JVM\n')
    
    # Start JVM
    ij = imagej.init('sc.fiji:fiji:2.1.1+net.imagej:imagej-legacy:0.37.4', headless=True)
    
    # Retreive all available operations from pyimagej
    #imagej_help_docs = scyjava.to_python(ij.op().help())
    #print(imagej_help_docs)
    
    print('Parsing imagej op help\n')
    
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
    plugin_path = base_path.joinpath('cookietin')

    # Get path to cookiecutter.json file that lives in each of the 40 folders within cookietin
    cookiecutter_path = base_path.joinpath('cookiecutter.json')

    # Get list of all plugin directories in the cookietin directory 
    plugins = list(plugin_path.iterdir()) 

    # Iterate over all plugin directories in the cookietin directory 
    for i,plugin in enumerate(reversed(plugins)):
        
        if plugin.name == "image-integral":
            
            # Create a path for the plugin
            path = Path(os.getcwd()).joinpath('polus-imagej-' + plugin.name + '-plugin')
            print(path)
            
            # If the pluging path is already a directory remove it and its children
            if path.is_dir():
                shutil.rmtree(str(path.absolute()))
        
            # Move the cookiecutter.json file for the current plug in to the polus-imagej-util directory overwriting last plugin json file
            shutil.copy(str(plugin.joinpath('cookiecutter.json')), cookiecutter_path)
            
            # Run the cookiecutter utility for the plugin
            os.system('cookiecutter ./utils/polus-imagej-util/ --no-input')
        
            break
        
        else:
            pass

