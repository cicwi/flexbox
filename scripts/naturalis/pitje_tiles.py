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

%pylab

#%% Run a test:
    
path = '/export/scratch2/kostenko/archive/Natrualis/pitje/femur/high_res/femur_batch/block_2/stack_1'
proj, meta = flex.compute.process_flex(path, options = {'bin':4, 'memmap': None}) 
vol = flex.project.init_volume(proj)
flex.project.FDK(proj, vol, meta['geometry'])
flex.util.display_slice(vol, dim = 0,title = 'FDK')    

path = '/export/scratch2/kostenko/archive/Natrualis/pitje/femur/high_res/femur_batch/block_2/stack_2'
proj1, meta1 = flex.compute.process_flex(path, options = {'bin':4, 'memmap': None}) 
vol1 = flex.project.init_volume(proj1)
flex.project.FDK(proj1, vol1, meta1['geometry'])
flex.util.display_slice(vol1, dim = 0,title = 'FDK')    

#%% merge:
input_paths = ['/export/scratch2/kostenko/archive/Natrualis/pitje/femur/high_res/femur_batch/block_1/stack_1', 
'/export/scratch2/kostenko/archive/Natrualis/pitje/femur/high_res/femur_batch/block_1/stack_2']
output_path = '/export/scratch2/kostenko/archive/Natrualis/pitje/femur/high_res/femur_batch/block_1/FDK_'

total, meta = merge_projections(input_paths)

flex.util.display_slice(total, dim = 1)

vol = flex.project.init_volume(total)
flex.project.FDK(total, vol, meta['geometry'])

flex.util.display_slice(vol, dim = 0,title = 'FDK')  

vol = flex.data.cast2type(vol, 'uint8', [0, 1])
flex.data.write_raw(output_path, 'vol', vol, dim = 0)
flex.data.write_meta(output_path + 'meta.toml', meta)   

#%% Define things:
    
# Define path:    
paths_a = []
paths_b = []
paths_c = []
paths_d = []
paths_e = []
paths_f = []

paths_a.append('/export/scratch2/kostenko/archive/Natrualis/pitje/femur/high_res/original_orientation/t1')
paths_a.append('/export/scratch2/kostenko/archive/Natrualis/pitje/femur/high_res/original_orientation/t2')
output_path_a = '/export/scratch2/kostenko/archive/Natrualis/pitje/femur/high_res/original_orientation/FDK_1/'

paths_b.append('/export/scratch2/kostenko/archive/Natrualis/pitje/femur/high_res/original_orientation/t3')
paths_b.append('/export/scratch2/kostenko/archive/Natrualis/pitje/femur/high_res/original_orientation/t4')
output_path_b = '/export/scratch2/kostenko/archive/Natrualis/pitje/femur/high_res/original_orientation/FDK_2/'

paths_c.append('/export/scratch2/kostenko/archive/Natrualis/pitje/femur/high_res/original_orientation/t5')
paths_c.append('/export/scratch2/kostenko/archive/Natrualis/pitje/femur/high_res/original_orientation/t6')
output_path_c = '/export/scratch2/kostenko/archive/Natrualis/pitje/femur/high_res/original_orientation/FDK_3/'

paths_d.append('/export/scratch2/kostenko/archive/Natrualis/pitje/femur/high_res/original_orientation/t7')
paths_d.append('/export/scratch2/kostenko/archive/Natrualis/pitje/femur/high_res/original_orientation/t8')
output_path_d = '/export/scratch2/kostenko/archive/Natrualis/pitje/femur/high_res/original_orientation/FDK_4/'

paths_e.append('/export/scratch2/kostenko/archive/Natrualis/pitje/femur/high_res/original_orientation/t9')
paths_e.append('/export/scratch2/kostenko/archive/Natrualis/pitje/femur/high_res/original_orientation/t10')
output_path_e = '/export/scratch2/kostenko/archive/Natrualis/pitje/femur/high_res/original_orientation/FDK_5/'

paths_f.append('/export/scratch2/kostenko/archive/Natrualis/pitje/femur/high_res/original_orientation/t11')
paths_f.append('/export/scratch2/kostenko/archive/Natrualis/pitje/femur/high_res/original_orientation/t12')
output_path_f = '/export/scratch2/kostenko/archive/Natrualis/pitje/femur/high_res/original_orientation/FDK_6/'

paths_af.append('/export/scratch2/kostenko/archive/Natrualis/pitje/femur/high_res/flipped_orientation/t1')
paths_af.append('/export/scratch2/kostenko/archive/Natrualis/pitje/femur/high_res/flipped_orientation/t2')
output_path_af = '/export/scratch2/kostenko/archive/Natrualis/pitje/femur/high_res/flipped_orientation/FDK_1/'

paths_bf.append('/export/scratch2/kostenko/archive/Natrualis/pitje/femur/high_res/flipped_orientation/t3')
paths_bf.append('/export/scratch2/kostenko/archive/Natrualis/pitje/femur/high_res/flipped_orientation/t4')
output_path_bf = '/export/scratch2/kostenko/archive/Natrualis/pitje/femur/high_res/flipped_orientation/FDK_2/'

paths_bf.append('/export/scratch2/kostenko/archive/Natrualis/pitje/femur/high_res/flipped_orientation/t5')
paths_bf.append('/export/scratch2/kostenko/archive/Natrualis/pitje/femur/high_res/flipped_orientation/t6')
output_path_bf = '/export/scratch2/kostenko/archive/Natrualis/pitje/femur/high_res/flipped_orientation/FDK_3/'

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
        proj, meta = flex.compute.process_flex(path, options = {'bin':bins, 'memmap': None})  
        
        # Correct beam hardeinng:
        #proj = flex.spectrum.equivalent_density(proj, meta['geometry'], energy, spec, compound = 'Al', density = 2.7)     
    
        flex.compute.append_tile(proj, meta['geometry'], total, tot_geom)
        
        #flex.util.display_slice(total, dim = 1)
        
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
vol = flex.data.cast2type(vol, 'uint8', [0, 100])
flex.data.write_raw(output_path_a, 'vol', vol, dim = 0)
flex.data.write_meta(output_path_a + 'meta.toml', meta)    

# B
    
total, tot_geom = merge_projections(paths_b)

vol = numpy.zeros([760, 1800, 1800], dtype = 'float32')
flex.project.FDK(total, vol, meta['geometry'])

vol = flex.data.cast2type(vol, 'uint8', [0, 100])
flex.data.write_raw(output_path_b, 'vol', vol, dim = 0)
flex.data.write_meta(output_path_b + 'meta.toml', meta)    

# C    

total, tot_geom = merge_projections(paths_c)

vol = numpy.zeros([760, 1800, 1800], dtype = 'float32')
flex.project.FDK(total, vol, meta['geometry'])

vol = flex.data.cast2type(vol, 'uint8', [0, 100])
flex.data.write_raw(output_path_c, 'vol', vol, dim = 0)
flex.data.write_meta(output_path_c + 'meta.toml', meta)    
    
# D   

total, tot_geom = merge_projections(paths_d)

vol = numpy.zeros([760, 1800, 1800], dtype = 'float32')
flex.project.FDK(total, vol, meta['geometry'])

vol = flex.data.cast2type(vol, 'uint8', [0, 100])
flex.data.write_raw(output_path_d, 'vol', vol, dim = 0)
flex.data.write_meta(output_path_d + 'meta.toml', meta)    

# E    

total, tot_geom = merge_projections(paths_e)

vol = numpy.zeros([760, 1800, 1800], dtype = 'float32')
flex.project.FDK(total, vol, meta['geometry'])

vol = flex.data.cast2type(vol, 'uint8', [0, 100])
flex.data.write_raw(output_path_e, 'vol', vol, dim = 0)
flex.data.write_meta(output_path_e + 'meta.toml', meta)    

# F    

total, tot_geom = merge_projections(paths_f)

vol = numpy.zeros([760, 1800, 1800], dtype = 'float32') 
flex.project.FDK(total, vol, meta['geometry'])

vol = flex.data.cast2type(vol, 'uint8', [0, 100])
flex.data.write_raw(output_path_f, 'vol', vol, dim = 0)
flex.data.write_meta(output_path_f + 'meta.toml', meta)    

# AF    

total, tot_geom = merge_projections(paths_af)

vol = numpy.zeros([760, 1800, 1800], dtype = 'float32') 
flex.project.FDK(total, vol, meta['geometry'])

vol = flex.util.cast2type(vol, 'uint8', [0, 100])
flex.data.write_raw(output_path_af, 'vol', vol, dim = 0)
flex.data.write_meta(output_path_af + 'meta.toml', meta)   

# BF    

total, tot_geom = merge_projections(paths_bf)

vol = numpy.zeros([760, 1800, 1800], dtype = 'float32') 
flex.project.FDK(total, vol, meta['geometry'])

vol = flex.util.cast2type(vol, 'uint8', [0, 100])
flex.data.write_raw(output_path_bf, 'vol', vol, dim = 0)
flex.data.write_meta(output_path_bf + 'meta.toml', meta)   

# CF    

total, tot_geom = merge_projections(paths_cf)

vol = numpy.zeros([760, 1800, 1800], dtype = 'float32') 
flex.project.FDK(total, vol, meta['geometry'])

vol = flex.util.cast2type(vol, 'uint8', [0, 100])
flex.data.write_raw(output_path_cf, 'vol', vol, dim = 0)
flex.data.write_meta(output_path_cf + 'meta.toml', meta)   