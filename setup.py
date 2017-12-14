import os
from setuptools import setup, find_packages


root_path = os.path.dirname(__file__)

# Determine version from top-level package __init__.py file
with open(os.path.join(root_path, 'flexbox', '__init__.py')) as f:
    for line in f:
        if line.startswith('__version__'):
            version = line.strip().split()[-1][1:-1]
            break

setup(
    name="flexbox",
    package_dir={'flexbox': 'flexbox'},
    packages=find_packages(),

    install_requires=[
        "transforms3d >= 0.3"],
    version=version,
)
