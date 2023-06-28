import os
import shutil
from pathlib import Path
import logging
from os import environ

logging.basicConfig(
    format="%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)
POLUS_LOG = getattr(logging, environ.get("POLUS_LOG", "DEBUG"))
logger = logging.getLogger("polus-python-template-post")
logger.setLevel(POLUS_LOG)

"""Create the final plugin directory structure.
(ex: polus/plugins/package1/package2/awesome_function)
and copy our generated plugin there.
"""
# rename the original source directory
os.rename("src", "src_template")
# create the new source directory
package_dir = "{{ cookiecutter.plugin_package }}"
package_dir = package_dir.split(".")
package_dir = os.path.join(*package_dir)

src = Path("src") / package_dir
os.makedirs(src, exist_ok=False)
# copy old source in new source structure
src_template = Path("src_template")
files = os.listdir(src_template)
for file_name in files:
    shutil.move(src_template / file_name, src)
# remove the original source directory
os.rmdir(src_template)

logger.debug(f"created source directories : {src}")

""" Buid the correct directories inside polus-plugins.
The directory structure must conforms to the plugin's spec :
    - dash-separated word in identifier.
    - folder hierarchy matches package namespace minus "polus.plugins"
    - plugin's folder name reflects the plugin package name but ends with "-plugin"
Ex: polus.plugins.package1.package2.awesome_function becomes
package1/package2/awesome-function-plugin
"""
# keep track of our new project
source_dir = Path(os.getcwd())

# try to find the project's root, otherwise we stay in the
# staging directory
final_dir = source_dir.parent
for folder in Path(final_dir).parent.parents:
    if os.path.exists(folder / ".git"):
        final_dir = folder
        break

# by default we create a plugin directory at the root    
target_dir = final_dir

# figure out if additional directories need to be created at the root
# make sure we replace underscores
new_dirs = "{{cookiecutter.plugin_package}}".replace("_", "-")
new_dirs = new_dirs.split(".")
# remove polus.plugins so we only keep intermediary directories
# Ex: polus.plugins.package1.package2.awesome_function creates
# package1/package2/
new_dirs = new_dirs[2:-1]
if len(new_dirs) != 0:
    package_dir = os.path.join(*new_dirs)
    target_dir = final_dir / package_dir

# create the plugin directory
os.makedirs(target_dir, exist_ok=True)
"""Move staged files to the the final target repo."""
print(f"moving sources from {source_dir} to {target_dir}")
shutil.move(source_dir, target_dir)
