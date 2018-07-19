#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Use Al dummy to callibrate density. Corrected reconstruction shoud show the density of aluminum = 2.7 g/cm3
"""
#%%
import flexbox as flex
import numpy

#%% Read

path = '/ufs/ciacc/flexbox/al_test/90KV_no_filt/'

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
    
energy, spectrum = flex.compute.calibrate_spectrum(proj, vol,  meta['geometry'], compound = 'Al', density = 2.7, n_bin = 21, iterations = 1000)   

# Save:
numpy.savetxt(path + 'spectrum.txt', [energy, spectrum]) 

#%% Test:

proj_ = flex.compute.equivalent_density(proj,  meta['geometry'], energy, spectrum, compound = 'Al', density = 2.7) 

vol = flex.project.init_volume(proj)
flex.project.FDK(proj_, vol, meta['geometry'])

#vol /= meta['geometry']['img_pixel'] ** 4

flex.util.display_slice(vol, title = 'Corrected FDK')

a,b = flex.compute.histogram(vol)
        
