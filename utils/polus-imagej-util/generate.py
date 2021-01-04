'''

This script generates plugin json manifests for all 40 IJ plugins, the names of which can be
found in the cookiecutter.json files within utils/polus-imagej-util/cookietin.

utils/polus-imagej-util/populate.py creates the .json dictionaries for each plugin,
so populate.py must be run prior to running this script

'''


from pathlib import Path
import os, shutil

##specify base path so that generate.py can access all 40 json files within cookietin
base_path = Path(__file__).parent
plugin_path = base_path.joinpath('cookietin')

##path to each cookiecutter.json file that lives in each of the 40 folders within cookietin
cookiecutter_path = base_path.joinpath('cookiecutter.json')

##iterate over all files in cookietin folder
plugins = list(plugin_path.iterdir()) 


for i,plugin in enumerate(reversed(plugins)):
    path = Path(os.getcwd()).joinpath('polus-imagej-' + plugin.name + '-plugin')
    #print(path)
    if path.is_dir():
        shutil.rmtree(str(path.absolute()))
    

    shutil.copy(str(plugin.joinpath('cookiecutter.json')),cookiecutter_path)
    
    os.system('cookiecutter ./utils/polus-imagej-util/ --no-input')
 