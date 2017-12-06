#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test flexData module.
"""
#%%
import flexData
import flexProject
import flexUtil
import flexCompute

import numpy

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

#%% Use optimize_rotation_center:
    
guess = flexCompute.optimize_rotation_center(proj, meta['geometry'], guess = 0, subscale = 16)

#%% Recon
meta['geometry']['axs_hrz'] = guess

vol = flexProject.inint_volume(proj)
flexProject.FDK(proj, vol, meta['geometry'])

flexUtil.display_slice(vol, bounds = [], title = 'FDK')


#%%


