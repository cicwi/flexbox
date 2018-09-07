#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test flex.data module.
"""
#%%
import flexbox as flex
import numpy

#%% Read

path = '/ufs/ciacc/flexbox/al_test/90KV_no_filt/'

dark = flex.data.read_raw(path, 'di')
flat = flex.data.read_raw(path, 'io')    
proj = flex.data.read_raw(path, 'scan_')

meta = flex.data.read_log(path, 'flexray')   
 
#%% Prepro:
    
proj = (proj - dark) / (flat.mean(0) - dark)
proj = -numpy.log(proj)

proj = flex.data.raw2astra(proj)    

flex.util.display_slice(proj, title = 'Sinogram')

#%% FDK Recon

vol = numpy.zeros([1, 2000, 2000], dtype = 'float32')

flex.project.FDK(proj, vol, meta['geometry'])

flex.util.display_slice(vol, bounds = [], title = 'FDK')

#%% Short implementation:

proj, meta = flex.compute.process_flex(path) 

vol = flex.project.init_volume(proj)
flex.project.FDK(proj, vol, meta['geometry'])
flex.util.display_slice(vol, dim = 0, title = 'FDK')    

#%% EM

vol = numpy.ones([10, 2000, 2000], dtype = 'float32')

flex.project.EM(proj, vol, meta['geometry'], iterations = 3)

flex.util.display_slice(vol, title = 'EM')

#%% SIRT with additional options
vol = numpy.zeros([1, 2000, 2000], dtype = 'float32')

options = {'bounds':[0, 1000], 'l2_update':True, 'block_number':3, 'mode':'sequential'}
flex.project.SIRT(proj, vol, meta['geometry'], iterations = 3, options = options)

flex.util.display_slice(vol, title = 'SIRT')