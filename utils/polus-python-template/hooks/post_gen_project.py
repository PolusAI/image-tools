import os
import shutil

print(os.getcwd())

def remove(filepath):
    if os.path.isfile(filepath):
        os.remove(filepath)
    elif os.path.isdir(filepath):
        shutil.rmtree(filepath)

create_log4j = '{{ cookiecutter.use_bfio }}'.lower()=='true'

if not create_log4j:
    # remove top-level file inside the generated folder
    remove('./src/log4j.properties')