#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test flexData module.
"""
#%%
import flexData
import flexProject
import flexUtil
import flexModel

import numpy

#%% Create volume and forward project:
    
vol = numpy.zeros([1, 512, 512], dtype = 'float32')
proj = numpy.zeros([1, 361, 512], dtype = 'float32')

geometry = flexData.empty_geometry([0, 360], 361)
geometry['']

flexModel.phantom(vol, 'sphere', [150, 15])    

flexProject.FDK(proj, vol, geometry)

#%% Read

path = '/export/scratch2/kostenko/archive/OwnProjects/al_tests/new/90KV_no_filt/'

dark = flexData.read_raw(path, 'di')
flat = flexData.read_raw(path, 'io')    
proj = flexData.read_raw(path, 'scan_')

geometry, p, l = flexData.read_log(path, 'flexray')   
 
#%% Prepro:
    
proj = (proj - dark) / (flat.mean(0) - dark)
proj = -numpy.log(proj)

proj = flexData.astra_projections(proj)    

flexUtil.display_slice(proj)

#%% Recon



flexProject.FDK(proj, vol, geometry)

flexUtil.display_slice(vol)
