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

#%% Read

path = '/export/scratch2/kostenko/archive/OwnProjects/al_tests/new/90KV_no_filt/'

dark = flexData.read_raw(path, 'di')
flat = flexData.read_raw(path, 'io')    
proj = flexData.read_raw(path, 'scan_', memmap = '/export/scratch3/kostenko/flexbox_swap/swap.prj')

meta = flexData.read_log(path, 'flexray')   
 
#%% Prepro:
    
# Now, since the data is on the harddisk, we shouldn't lose the pointer to it!    
# Be careful which operations to apply. Implicit are OK.
proj -= dark
proj /= (flat.mean(0) - dark)

numpy.log(proj, out = proj)
proj *= -1

proj = flexData.raw2astra(proj)    

flexUtil.display_slice(proj)

#%% Recon

vol = numpy.zeros([50, 2000, 2000], dtype = 'float32')

flexProject.FDK(proj, vol, meta['geometry'])

flexUtil.display_slice(vol)

#%% SIRT

vol = numpy.ones([50, 2000, 2000], dtype = 'float32')

options = {'block_number':10, 'index':'sequential'}
flexProject.SIRT(proj, vol, meta['geometry'], iterations = 5)

flexUtil.display_slice(vol, title = 'SIRT')