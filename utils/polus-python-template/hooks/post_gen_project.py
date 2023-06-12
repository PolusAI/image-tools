import os
import shutil
from pathlib import Path

os.rename("src", "src_template")  # rename the source template

"""Create the plugin's source directory structure"""
package_dir = "{{ cookiecutter.plugin_package }}"
package_dir = package_dir.split(".")
package_dir = os.path.join(*package_dir)
src = Path("src") / package_dir
if not os.path.exists(src):
    os.makedirs(src)

"""Copying src folder into the new directory structure"""
src_template = Path("src_template")
files = os.listdir(src_template)
for file_name in files:
    shutil.move(src_template / file_name, src)
os.rmdir(src_template)  # remove the source template

""" Build the plugin's directory structure in the polus-plugins repository.

The directory structure must conforms to the plugin's spec :
    - dash-separated word in identifier.
    - folder hierarchy matches package namespace minus the root polus package.
    - plugin's folder name reflects the plugin package name but ends with "-plugin"
"""
source_dir = Path(os.getcwd())
root_dir = source_dir.parent  # default to polus-python-template

package_dir = "{{cookiecutter.plugin_package}}"
package_dir = package_dir.split(".")
package_dir = package_dir[2:-1]  # remove polus and plugins from package's list
package_dir = os.path.join(*package_dir)

# find the project's root
for folder in Path(source_dir).parent.parents:
    if os.path.exists(folder / "pyproject.toml"):
        root_dir = folder
        break

plugin_slug = "{{cookiecutter.plugin_slug}}"
if plugin_slug.startswith("polus-"):
    plugin_dir = plugin_slug[6:]
else:
    plugin_dir = plugin_slug

target_dir = root_dir / package_dir / plugin_dir

"""Move staged files to the the final target repo."""
print(f"moving sources from {source_dir} to {target_dir}")
shutil.move(source_dir, target_dir)
