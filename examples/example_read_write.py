#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test reading Flexray raw and writing ASTRA readable
"""
#%%
import flexbox as flex
import sys

#%% Read / write a geometry file:
if len(sys.argv) == 2:
    path = sys.argv[1]
else:
    path = '/export/scratch2/kostenko/archive/OwnProjects/al_tests/new/90KV_no_filt/'

meta = flex.data.read_log(path, 'flexray') 

flex.data.write_meta(path + 'flexray.toml', meta)

#%% Read / write raw data files:
    
dark = flex.data.read_raw(path, 'di')
flat = flex.data.read_raw(path, 'io')    
proj = flex.data.read_raw(path, 'scan_')

#%% Read geometry and convert to ASTRA:

meta_1 = flex.data.read_meta(path + 'flexray.toml') 

vol_geom = flex.data.astra_vol_geom(meta['geometry'], [100, 100, 100])
proj_geom = flex.data.astra_proj_geom(meta['geometry'], proj.shape)
    
print(vol_geom)
print(proj_geom)
