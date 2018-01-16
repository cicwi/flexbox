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

#%% Define things:
    
# Define path:    
paths_a = []
paths_b = []
paths_c = []

paths_a.append('/export/scratch2/kostenko/archive/Natrualis/pitje/skull_cap/high_res/t1')
paths_a.append('/export/scratch2/kostenko/archive/Natrualis/pitje/skull_cap/high_res/t2')
output_path_a = '/export/scratch2/kostenko/archive/Natrualis/pitje/skull_cap/high_res/FDK_1'

paths_b.append('/export/scratch2/kostenko/archive/Natrualis/pitje/skull_cap/high_res/t3')
paths_b.append('/export/scratch2/kostenko/archive/Natrualis/pitje/skull_cap/high_res/t4')
output_path_b = '/export/scratch2/kostenko/archive/Natrualis/pitje/skull_cap/high_res/FDK_2'

paths_c.append('/export/scratch2/kostenko/archive/Natrualis/pitje/skull_cap/high_res/t5')
paths_c.append('/export/scratch2/kostenko/archive/Natrualis/pitje/skull_cap/high_res/t6')
output_path_c = '/export/scratch2/kostenko/archive/Natrualis/pitje/skull_cap/high_res/FDK_3'


# Load data needed for beam hardening correction:
energy, spec = numpy.loadtxt('/export/scratch2/kostenko/archive/OwnProjects/al_tests/new/90KV_1mm_brass/spectrum.txt')
        
#%% Read, process, merge, reconstruct data:
 
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
# A
total, meta = merge_projections(paths_a)
  
vol = numpy.zeros([760, 1800, 1800], dtype = 'float32')
flex.project.FDK(total, vol, meta['geometry'])

# Save reconstruction:    
vol = flex.util.cast2type(vol, 'uint8', [0, 10])
flex.data.write_raw(output_path_a, 'vol', vol, dim = 0)
flex.data.write_meta(output_path + 'meta.toml', meta)    

# B
    
total, tot_geom = merge_projections(paths_b)

vol *= 0 
flex.project.FDK(total, vol, meta['geometry'])

vol = flex.util.cast2type(vol, 'uint8', [0, 10])
flex.data.write_raw(output_path_b, 'vol', vol, dim = 0)

# C    
total, tot_geom = merge_projections(paths_c)

vol *= 0 
flex.project.FDK(total, vol, meta['geometry'])

vol = flex.util.cast2type(vol, 'uint8', [0, 10])
flex.data.write_raw(output_path_c, 'vol', vol, dim = 0)

    
