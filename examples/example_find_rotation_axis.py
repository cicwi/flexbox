#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test flex.data module.
"""
#%%
import flexbox as flex
import sys
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

#%% Use optimize_rotation_center:
    
guess = flex.compute.optimize_rotation_center(proj, meta['geometry'], guess = 0, subscale = 8)

#%% Recon
meta['geometry']['axs_hrz'] = guess

vol = flex.project.init_volume(proj)
flex.project.FDK(proj, vol, meta['geometry'])

flex.util.display_slice(vol, bounds = [], title = 'FDK')


#%%


