#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This example is using the skull data where each 2 tiles (horizontal) should be merged before FBP is applied
and after 3 FBP reconstructions are done, their results should be merged in volume space.
"""
#%%

import flexbox as flex

import numpy

#%% Define things:
    
# Define path:    
paths = []

paths.append('/export/scratch2/kostenko/archive/Natrualis/pitje/skull_cap/high_res/t3')
paths.append('/export/scratch2/kostenko/archive/Natrualis/pitje/skull_cap/high_res/t4')
output_path = '/export/scratch2/kostenko/archive/Natrualis/pitje/skull_cap/high_res/sirt'
    
#%% Inintialize and read the input:    
bins = 4
proj_shape = [768//2, 2000//2, 972//2]

# Load data needed for beam hardening correction:
energy, spec = numpy.loadtxt('/export/scratch2/kostenko/archive/OwnProjects/al_tests/new/90KV_1mm_brass/spectrum.txt')
    
# Read geometries:        
geoms = []    
projs = []

for path in paths: 
    # meta:
    proj, meta = flex.compute.process_flex(path, options = {'bin':bins, 'disk_map': None}) 
    
    # Correct beam hardeinng:
    proj = flex.spectrum.equivalent_density(proj, meta['geometry'], energy, spec, compound = 'Al', density = 2.7)     
    
    # TODO: fill in thetas properly
    meta['geometry']['thetas'] = numpy.linspace(0, 360, proj.shape[1], dtype = 'float32')
    
    projs.append(proj)
    geoms.append(meta['geometry'])
    
#%% Reconstruct:
    
vol = numpy.zeros([330, 1800//2, 1800//2], dtype = 'float32')

options = {'bounds':[0, 1000], 'l2_update':True, 'block_number':10, 'index':'sequential'}
flex.project.SIRT_tiled(projs, vol, geoms, iterations = 5, options = options)
  
flex.util.display_slice(vol, title = 'SIRT')

#%% Save reconstruction:    
vol = flex.util.cast2type(vol, 'uint8', [0, 10])
flex.data.write_raw(output_path, 'vol', vol, dim = 0)
flex.data.write_meta(output_path + 'meta.toml', meta)    