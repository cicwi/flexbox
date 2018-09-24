import os
from setuptools import setup, find_packages

setup(
    name="flexbox",
    package_dir={'flexbox': 'flexbox'},
    packages=find_packages(),

    install_requires=[
    "numpy>=1.0",
    "scipy>=0.14"
    "scikit-image>=0.13.0",
    "paramiko",
    "scp",	
    #"matplotlib>=2.0.0", # these are problematic packages somehow...
    #"transforms3d",
    #"imageio>=2.2.0",
    #"astra-toolbox>1.8.3",
    #"xraylib>=3.3.0",   # xraylib's conda package isn't found by setuptools
    "toml>=0.9.0"],

    version='0.0.1',
)
