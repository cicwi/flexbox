#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Use Al dummy to callibrate density.
"""
#%%
import flexbox as flex
import numpy

#%% Read

#path = '/export/scratch2/kostenko/archive/OwnProjects/al_tests/new/90KV_no_filt/'
path = '/export/scratch2/kostenko/archive/OwnProjects/al_tests/new/90KV_1mm_brass/'

dark = flex.data.read_raw(path, 'di')
flat = flex.data.read_raw(path, 'io')    
proj = flex.data.read_raw(path, 'scan_')

meta = flex.data.read_log(path, 'flexray')   
 
#%% Prepro:
    
proj = (proj - dark) / (flat.mean(0) - dark)
proj = -numpy.log(proj)
proj = flex.data.raw2astra(proj)    

proj = flex.compute.subtract_air(proj)

flex.util.display_slice(proj, title = 'Sinogram')

#%% Reconstruct:
    
vol = flex.project.init_volume(proj)
flex.project.FDK(proj, vol, meta['geometry'])

flex.util.display_slice(vol, title = 'Uncorrected FDK')
    
energy, spectrum = flex.spectrum.calibrate_spectrum(proj, vol,  meta['geometry'], compound = 'Al', density = 2.7, n_bin = 21, iterations = 1000)   

# Save:
numpy.savetxt(path + 'spectrum.txt', [energy, spectrum]) 

#%% Test:

proj_ = flex.spectrum.equivalent_density(proj,  meta['geometry'], energy, spectrum, compound = 'Al', density = 2.7) 

vol = flex.project.init_volume(proj)
flex.project.FDK(proj_, vol, meta['geometry'])

vol /= meta['geometry']['img_pixel'] ** 4

flex.util.display_slice(vol, title = 'Corrected FDK')

flex.compute.histogram(vol)

#%% Test 2: different data

path = '/export/scratch2/kostenko/archive/Natrualis/pitje/femur/al_callibration/'
dark = flex.data.read_raw(path, 'di')
flat = flex.data.read_raw(path, 'io')    
proj = flex.data.read_raw(path, 'scan_')

meta = flex.data.read_log(path, 'flexray') 
    
proj = (proj - dark) / (flat.mean(0) - dark)
proj = -numpy.log(proj)
proj = flex.data.raw2astra(proj)   

flex.util.display_slice(proj, title = 'Bedore')

proj = flex.spectrum.equivalent_density(proj,  meta['geometry'], energy, spectrum, compound = 'Al', density = 2.7)  

flex.util.display_slice(proj, title = 'After')

vol = flex.project.init_volume(proj)
#flexProject.FDK(proj, vol, meta['geometry'])
flex.project.SIRT(proj, vol, meta['geometry'], iterations = 10)

flex.util.display_slice(vol / meta['geometry']['img_pixel'] ** 4, title = 'Corrected FDK')
  
     
    
