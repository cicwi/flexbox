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
import astra

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

#%% Recon:

vol = numpy.zeros([50, 2000, 2000], dtype = 'float32')

# Initialize ASTRA geometries:
vol_geom = flexProject.get_vol_geom(vol.shape, geometry)
proj_geom = flexProject.get_proj_geom(geometry, proj.shape[::2])
        
# This is ASTRAAA!!!
sin_id = astra.data3d.link('-sino', proj_geom, proj)

vol_id = astra.data3d.link('-vol', vol_geom, vol)

cfg = astra.astra_dict('FDK_CUDA')
cfg['ReconstructionDataId'] = vol_id
cfg['ProjectionDataId'] = sin_id

alg_id = astra.algorithm.create(cfg)
astra.algorithm.run(alg_id, 1)
  
astra.algorithm.delete(alg_id)
astra.data3d.delete(sin_id)
astra.data3d.delete(vol_id)

#%% Display:
    
flexUtil.display_slice(vol)    