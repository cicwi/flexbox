#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test reading Flexray raw and writing ASTRA readable
"""
#%%
import flexbox as flex

#%% Read / write a geometry file:

path = '/export/scratch2/kostenko/archive/OwnProjects/al_tests/new/90KV_no_filt/'

#path = 'D:\\Data\\al_dummy_vertical_tile_1\\'

meta = flex.data.read_log(path, 'flexray') 

flex.data.write_meta(path + 'flexray.toml', meta)

#%% Read / write raw data files:
    
dark = flex.data.read_raw(path, 'di')
flat = flex.data.read_raw(path, 'io')    
proj = flex.data.read_raw(path, 'scan_')

#%% Read geometry and convert to ASTRA:

meta_1 = flex.data.read_meta(path + 'flexray.toml') 

vol_geom = flex.data.astra_vol_geom(meta['geometry'], [100, 100, 100])
proj_geom = flex.data.astra_proj_geom(meta['geometry'], proj.shape[::2])
    
print(vol_geom)
print(proj_geom)
