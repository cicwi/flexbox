#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test flexData module.
"""
#%%
import flexData
import flexProject
import flexUtil
import numpy
import pycuda

#%% Read

path = '/export/scratch2/kostenko/archive/OwnProjects/al_tests/new/90KV_no_filt/'

dark = flexData.read_raw(path, 'di')
flat = flexData.read_raw(path, 'io')    
proj = flexData.read_raw(path, 'scan_')

meta = flexData.read_log(path, 'flexray')   
 
#%% Prepro:
    
proj = (proj - dark) / (flat.mean(0) - dark)
proj = -numpy.log(proj)

proj = flexData.raw2astra(proj)    

flexUtil.display_slice(proj, title = 'Sinogram')

#%% Recon

vol = numpy.zeros([1, 2000, 2000], dtype = 'float32')

flexProject.FDK(proj, vol, meta['geometry'])

flexUtil.display_slice(vol, bounds = [], title = 'FDK')

#%% EM

vol = numpy.ones([50, 2000, 2000], dtype = 'float32')

flexProject.EM(proj, vol, meta['geometry'], iterations = 5)

flexUtil.display_slice(vol, title = 'EM')

#%% SIRT
vol = numpy.zeros([1, 2000, 2000], dtype = 'float32')



options = {'bounds':[0, 1000], 'l2_update':True, 'block_number':1, 'index':'sequential'}
flexProject.SIRT(proj, vol, meta['geometry'], iterations = 1, options = options)

flexUtil.display_slice(vol, title = 'SIRT')

#%%
vol = numpy.ones([50, 2000, 2000], dtype = 'float32')

flexProject.backproject(proj, vol, meta['geometry'])
flexUtil.display_slice(vol, title = 'BP')

#%%
import flexCompute

#flexCompute._modifier_l2cost_(proj, meta['geometry'], [10, 8], -.0, 'axs_hrz', True)


flexCompute.optimize_rotation_center(proj, meta['geometry'], guess = 0, subscale = 4, center_of_mass = False)
