#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This example is using the skull data where each 2 tiles (horizontal) should be merged before FBP is applied
and after 3 FBP reconstructions are done, their results should be merged in volume space.
"""
#%%

import flexbox as flex

import numpy
import gc
        
#%% Read, process, merge, reconstruct data:

# Load data needed for beam hardening correction:
energy, spec = numpy.loadtxt('/export/scratch2/kostenko/archive/OwnProjects/al_tests/new/90KV_1mm_brass/spectrum.txt')
    
    
def merge_projections(input_paths):
    '''
    Merge datasets and reconstruct the total:
    '''
    bins = 2
    proj_shape = [768, 2000, 972]
        
    # Read geometries:    
        
    geoms = []    
    for path in input_paths: 
        # meta:
        meta = flex.data.read_log(path, 'flexray', bins = bins) 
        geoms.append(meta['geometry'])
                    
    # Initialize the total data based on all geometries and a single projection stack shape:  
    tot_shape, tot_geom = flex.data.tiles_shape(proj_shape, geoms)     
    
    total = numpy.memmap('/export/scratch3/kostenko/flexbox_swap/swap.prj', dtype='float32', mode='w+', shape = (tot_shape[0],tot_shape[1],tot_shape[2]))    
    
    # Read data:
    for path in input_paths: 
        # read and process:
        proj, meta = flex.compute.process_flex(path, options = {'bin':bins, 'disk_map': None})  
        
        # Correct beam hardeinng:
        proj = flex.spectrum.equivalent_density(proj, meta['geometry'], energy, spec, compound = 'Al', density = 2.7)     
    
        flex.data.append_tile(proj, meta['geometry'], total, tot_geom)
        
        flex.util.display_slice(total, dim = 1)
        
        # Free memory:
        del proj
        gc.collect()
        
    # TODO: fill in thetas properly
    tot_geom['thetas'] = numpy.linspace(0, 360, total.shape[1], dtype = 'float32')
    
    # Reaplce the geometry record in meta:
    meta['geometry'] = tot_geom
    
    return total, meta 

#%% Reconstruct:
    
# Define path:    
paths = []

paths.append('/export/scratch2/kostenko/archive/Natrualis/pitje/skull_cap/high_res/t3')
paths.append('/export/scratch2/kostenko/archive/Natrualis/pitje/skull_cap/high_res/t4')
output_path = '/export/scratch2/kostenko/archive/Natrualis/pitje/skull_cap/high_res/vol_1'

total, meta = merge_projections(paths)
  
vol = numpy.zeros([760, 1800, 1800], dtype = 'float32')
flex.project.FDK(total, vol, meta['geometry'])

# Save reconstruction:    
vol = flex.util.cast2type(vol, 'uint8', [0, 10])
flex.data.write_raw(output_path, 'vol', vol, dim = 0)
flex.data.write_meta(output_path + 'meta.toml', meta)    