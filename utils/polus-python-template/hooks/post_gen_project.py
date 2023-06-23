import os
import shutil
from pathlib import Path

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

# figure out the directories that need to be created at the root
# make sure we replace underscores
package_dir = "{{cookiecutter.plugin_package}}".replace("_", "-")
package_dir = package_dir.split(".")
# remove polus.plugins so we only keep intermediary directories
# Ex: polus.plugins.package1.package2.awesome_function creates
# package1/package2/
package_dir = package_dir[2:-1]
package_dir = os.path.join(*package_dir)
target_dir = final_dir / package_dir
os.makedirs(target_dir, exist_ok=False)

"""Move staged files to the the final target repo."""
print(f"moving sources from {source_dir} to {target_dir}")
shutil.move(source_dir, target_dir)
