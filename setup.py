from setuptools import setup, find_packages

# with open("README.md", "r") as fh:
#     long_description = fh.read()

with open("./polus/_plugins/VERSION", "r") as fh:
    version = fh.read()
    with open("./polus/_plugins/VERSION", "w") as fw:
        fw.write(version)

package_data = ["_plugins/VERSION", "manifests/*"]

setup(
    name="polus-plugins",
    version=version,
    author="Nick Schaub",
    author_email="nick.schaub@nih.gov",
    description="API for Polus Plugins.",
    # long_description=long_description,
    # long_description_content_type="text/markdown",
    # project_urls={
    #     'Documentation': 'https://bfio.readthedocs.io/en/latest/',
    #     'Source': 'https://github.com/polusai/polus-data'
    # },
    # entry_points={'napari.plugin': 'bfio = bfio.bfio'},
    packages=find_packages(),
    package_data={"polus": package_data},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "pygithub>=1.55",
        "docker>=5.0.3",
        "pydantic>=1.8.2",
        "python_on_whales>=0.34.0",
        "alive-progress>=2.1.0",
    ],
)
