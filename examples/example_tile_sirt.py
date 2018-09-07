#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This example is using SIRT applied to the skull data of 2 tiles.
"""
#%%

import flexbox as flex
import numpy

#%% Define things:
    
# Define path:    
paths = []

paths.append('/ufs/ciacc/SeeThroughMuseum/naturalis/dubois_collection/dec_2017/skull_cap/high_res/t1/')
paths.append('/ufs/ciacc/SeeThroughMuseum/naturalis/dubois_collection/dec_2017/skull_cap/high_res/t2/')
   
#%% Inintialize and read the input:    

geoms = []    
projs = []

binz = 4

for path in paths: 
    # meta:
    proj, meta = flex.compute.process_flex(path, options = {'bin':binz, 'disk_map': None}) 
    
    projs.append(proj)
    geoms.append(meta['geometry'])
    
#%% Reconstruct:
    
vol = numpy.zeros([30, 1800//binz, 1800//binz], dtype = 'float32')

options = {'bounds':[0, 1000], 'l2_update':True, 'block_number':10, 'index':'sequential'}
flex.project.SIRT_tiled(projs, vol, geoms, iterations = 5, options = options)
  
flex.util.display_slice(vol, title = 'SIRT')

