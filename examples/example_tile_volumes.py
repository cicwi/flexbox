#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This example is using the skull data where each 2 tiles (horizontal) should be merged before FBP is applied
and after 3 FBP reconstructions are done, their results should be merged in volume space.
"""
#%%

import flexbox as flex

import numpy
        
#%% Read, process, merge, reconstruct data:
    
def merge_volumes(path, tile, num):    

    import matplotlib.pyplot as plt
    import os                  # file-name routines
    
    path_ = path + tile%0

    meta = flex.data.read_meta(path_ + 'meta.toml')   

    vol_pos0 = meta['geometry']['vol_vrt']
    
    indexes = []
    vol_pos = []

    for ii in range(num):
        
        path_ = path + tile%ii
    
        meta = flex.data.read_meta(path_ + 'meta.toml')    
    
        vol_pos = meta['geometry']['vol_vrt']
    
        print(vol_pos)
    
        offset = vol_pos - vol_pos0
    
        offset = flex.data.mm2pixel(offset, meta['geometry'])
            
        index = numpy.int32(numpy.round(numpy.arange(0, 750) + offset))
        indexes.append(index)  
        
    for ii in range(0, numpy.max(indexes)):
                
        slices = []
    
        # Loop over volumes:
        for jj in range(num):
            index = indexes[jj]
            
            if ii in index:
                # Read volume slice
                            
                key = numpy.where(index == ii)[0][0]
                
                filename = (path + tile)%jj + 'vol_%06u.tiff'% key
                
                print('Reading:', filename)
                img = flex.data._read_tiff_(filename)
                
                slices.append(img)
                
        # Merge slices:
        if slices == []:
            break
        
        else:
            img = numpy.max(slices, 0)
            
            path_ = path + 'full_fdk/' 
            if not os.path.exists(path_):
                os.makedirs(path_)
                       
            plt.imshow(img)
            plt.show()
            
            filename = path_ + 'vol_%06u.tiff'% ii
            
            print('Writing:', filename)
            flex.data.write_tiff(filename, img)        

    
#%% Merge:    
merge_volumes('/export/scratch2/kostenko/archive/Natrualis/pitje/skull_cap/high_res/', 'vol_%u/', 3)      

    
