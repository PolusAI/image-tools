import setuptools

with open("VERSION",'r') as fh:
    version = fh.read()

setuptools.setup(
    name="Feature Extraction",
    version=version,
    author="Jayapriya Nagarajan",
    author_email="jayapriya.nagarajan@labshare.org",
    description="Calculate feret diameter,number of neighbors touching the object, polygonality and hexagonality of an object",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

