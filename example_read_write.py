#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test reading Flexray raw and writing ASTRA readable
"""
#%%
import flexData

#%% Read / write a geometry file:

#path = '/export/scratch2/kostenko/archive/OwnProjects/al_tests/new/90KV_no_filt/'

path = 'D:\\Data\\al_dummy_vertical_tile_1\\'

meta = flexData.read_log(path, 'flexray') 

flexData.write_meta(path + 'flexray.toml', meta)

#%% Read / write raw data files:
    
dark = flexData.read_raw(path, 'di')
flat = flexData.read_raw(path, 'io')    
proj = flexData.read_raw(path, 'scan_')

#%% Read geometry and convert to ASTRA:

meta_1 = flexData.read_meta(path + 'flexray.toml') 

vol_geom = flexData.astra_vol_geom(meta['geometry'], [100, 100, 100])
proj_geom = flexData.astra_proj_geom(meta['geometry'], proj.shape[::2])
    
print(vol_geom)
print(proj_geom)
