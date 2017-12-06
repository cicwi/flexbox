#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test flexData module. CUPY test.
"""
#%%
import flexData
import flexProject
import flexUtil
import numpy
import cupy

#%% Read

path = '/export/scratch2/kostenko/archive/OwnProjects/al_tests/new/90KV_no_filt/'

dark = flexData.read_raw(path, 'di')
flat = flexData.read_raw(path, 'io')    
proj = flexData.read_raw(path, 'scan_')

meta = flexData.read_log(path, 'flexray')   
 
#%% Prepro:
    
# Convert to CUPY:    
proj = cupy.array(proj)
flat = cupy.array(flat)
dark = cupy.array(dark)
    
# Use CUDA to compute stuff:
proj = (proj - dark) / (flat.mean(0) - dark)
proj = -cupy.log(proj)

proj = flexData.raw2astra(proj)    

flexUtil.display_slice(proj, title = 'Sinogram')

#%% Recon

vol = numpy.zeros([1, 2000, 2000], dtype = 'float32')

flexProject.FDK(proj, vol, meta['geometry'])

flexUtil.display_slice(vol, bounds = [], title = 'FDK')
