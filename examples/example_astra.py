#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test flex.data module.
"""
#%%
import flexbox as flex
import numpy
import astra
import sys

#%% Read data:
    
if len(sys.argv) == 2:
    path = sys.argv[1]
else:
    path = '/export/scratch3/kostenko/Fast_Data/salt_no_filter'

dark = flex.data.read_raw(path, 'di')
flat = flex.data.read_raw(path, 'io')    
proj = flex.data.read_raw(path, 'scan_')

meta = flex.data.read_log(path, 'flexray')   
 
#%% Prepro:
    
proj = (proj - dark) / (flat.mean(0) - dark)
proj = -numpy.log(proj)

proj = flex.data.raw2astra(proj)    

flex.util.display_slice(proj)

#%% Recon:

vol = numpy.zeros([50, 2000, 2000], dtype = 'float32')

# Initialize ASTRA geometries:
vol_geom = flex.data.astra_vol_geom(meta['geometry'], vol.shape)
proj_geom = flex.data.astra_proj_geom(meta['geometry'], proj.shape)
        
# This is ASTRAAA!!!
sin_id = astra.data3d.link('-sino', proj_geom, numpy.ascontiguousarray(proj))
vol_id = astra.data3d.link('-vol', vol_geom, numpy.ascontiguousarray(vol))

cfg = astra.astra_dict('FDK_CUDA')
cfg['ReconstructionDataId'] = vol_id
cfg['ProjectionDataId'] = sin_id

alg_id = astra.algorithm.create(cfg)
astra.algorithm.run(alg_id, 1)
  
astra.algorithm.delete(alg_id)
astra.data3d.delete(sin_id)
astra.data3d.delete(vol_id)

#%% Display:
    
flex.util.display_slice(vol)    
