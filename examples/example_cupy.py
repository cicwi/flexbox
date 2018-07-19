#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test flex.data module. CUPY test.
"""
#%%
import flexbox as flex
import numpy
import cupy

# OUT OF ORDER!!!

'''
#%% Read

path = '/export/scratch2/kostenko/archive/OwnProjects/al_tests/new/90KV_no_filt/'

dark = flex.data.read_raw(path, 'di')
flat = flex.data.read_raw(path, 'io')    
proj = flex.data.read_raw(path, 'scan_')

meta = flex.data.read_log(path, 'flexray')   
 
#%% Prepro:
    
# Convert to CUPY:    
proj = cupy.array(proj)
flat = cupy.array(flat)
dark = cupy.array(dark)
    
# Use CUDA to compute stuff:
proj = (proj - dark) / (flat.mean(0) - dark)
proj = -cupy.log(proj)

proj = flex.data.raw2astra(proj)    

flex.util.display_slice(proj, title = 'Sinogram')

#%% Recon

vol = numpy.zeros([1, 2000, 2000], dtype = 'float32')

flex.project.FDK(proj, vol, meta['geometry'])

flex.util.display_slice(vol, bounds = [], title = 'FDK')
'''