#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test flex.data module.
"""
#%%
import flexbox as flex
import numpy

#%% Read

path = '/export/scratch2/kostenko/archive/OwnProjects/al_tests/new/90KV_no_filt/'

dark = flex.data.read_raw(path, 'di')
flat = flex.data.read_raw(path, 'io')    
proj = flex.data.read_raw(path, 'scan_')

meta = flex.data.read_log(path, 'flexray')   
 
#%% Prepro:
    
proj = (proj - dark) / (flat.mean(0) - dark)
proj = -numpy.log(proj)

proj = flex.data.raw2astra(proj)    

flex.util.display_slice(proj, title = 'Sinogram')

#%% Recon

vol = numpy.zeros([1, 2000, 2000], dtype = 'float32')

flex.project.FDK(proj, vol, meta['geometry'])

flex.util.display_slice(vol, bounds = [], title = 'FDK')

#%% EM

vol = numpy.ones([50, 2000, 2000], dtype = 'float32')

flex.project.EM(proj, vol, meta['geometry'], iterations = 5)

flex.util.display_slice(vol, title = 'EM')

#%% SIRT
vol = numpy.zeros([1, 2000, 2000], dtype = 'float32')

options = {'bounds':[0, 1000], 'l2_update':True, 'block_number':1, 'index':'sequential'}
flex.project.SIRT(proj, vol, meta['geometry'], iterations = 1, options = options)

flex.util.display_slice(vol, title = 'SIRT')

#%% Short implementation:
    
proj, meta = flex.compute.process_flex('/export/scratch2/kostenko/archive/OwnProjects/al_tests/new/90KV_no_filt/', options = {'bin':1, 'memmap': None}) 

vol = flex.project.init_volume(proj)
flex.project.FDK(proj, vol, meta['geometry'])
flex.util.display_slice(vol, dim = 0, title = 'FDK')    