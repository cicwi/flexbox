# Installation instructions...

If you intend to install Flexbox as a conda package, use:

> conda install -c teascavenger -c conda-forge -c astra-toolbox/label/dev -c owlas flexbox

This will install dependencies from conda-forge and owlas channels.

If you want to use pip and setuptools (setup.py), first install all of the dependecies:

> conda install scikit-image
> conda install matplotlib

> conda install -c astra-toolbox/label/dev astra-toolbox

> conda install -c conda-forge tifffile
> conda install -c conda-forge xraylib=3.3.0
> conda install -c conda-forge toml

> conda install -c owlas transforms3d

> git clone https://github.com/cicwi/flexbox
> cd flexbox

> pip install -e .

