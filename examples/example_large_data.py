#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test flexData module.
"""
#%%
import flexbox as flex
import numpy

#%% Read

path = '/ufs/ciacc/flexbox/al_test/90KV_no_filt/'

dark = flex.data.read_raw(path, 'di')
flat = flex.data.read_raw(path, 'io')    
proj = flex.data.read_raw(path, 'scan_', memmap = '/ufs/ciacc/flexbox/swap/swap.prj')

meta = flex.data.read_log(path, 'flexray')   
 
#%% Prepro:
    
# Now, since the data is on the harddisk, we shouldn't lose the pointer to it!    
# Be careful which operations to apply. Implicit are OK.
proj -= dark
proj /= (flat.mean(0) - dark)

numpy.log(proj, out = proj)
proj *= -1

proj = flex.data.raw2astra(proj)    

flex.util.display_slice(proj)

#%% Recon

vol = numpy.zeros([50, 2000, 2000], dtype = 'float32')

flex.project.FDK(proj, vol, meta['geometry'])

flex.util.display_slice(vol)

#%% SIRT

vol = numpy.ones([50, 2000, 2000], dtype = 'float32')

options = {'block_number':10, 'index':'sequential'}
flex.project.SIRT(proj, vol, meta['geometry'], iterations = 5)

flex.util.display_slice(vol, title = 'SIRT')