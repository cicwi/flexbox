#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script is made to produce ASTRA projection geometry that can be used by matlab code, for instance.
Args:
    path (str): path to the log file.
    data_shape (list(3)): dimensions of the projection data
"""
import sys
import os
import flexbox as flex

# Extract arguments:
path = sys.argv[1]
data_shape = sys.argv[2]

print('Reading log file at:', path)
print('Data shape', data_shape)

# Read:
meta = flex.data.read_log(path, 'flexray')   

# Write:
flex.data.write_meta(os.path.join(path, 'flexray.toml'), meta)
flex.data.write_astra(os.path.join(path, 'projection.geom'), data_shape, meta['geometry'])

