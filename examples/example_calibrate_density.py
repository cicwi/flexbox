#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulate spectral data with Poisson noise
"""
#%%
import flexData
import flexProject
import flexUtil

import numpy

#%% Read

#path = '/export/scratch2/kostenko/archive/OwnProjects/al_tests/new/90KV_no_filt/'
path = '/export/scratch2/kostenko/archive/OwnProjects/al_tests/new/90KV_1mm_brass/'
dark = flexData.read_raw(path, 'di')
flat = flexData.read_raw(path, 'io')    
proj = flexData.read_raw(path, 'scan_')

meta = flexData.read_log(path, 'flexray')   
 
#%% Prepro:
    
proj = (proj - dark) / (flat.mean(0) - dark)
proj = -numpy.log(proj)

proj = flexData.raw2astra(proj)    

flexUtil.display_slice(proj, title = 'Sinogram')

#%% Reconstruct:
    
vol = flexProject.init_volume(proj)
flexProject.FDK(proj, vol, meta['geometry'])

vol /= meta['geometry']['img_pixel'] ** 4
flexUtil.display_slice(vol, title = 'Uncorrected FDK')
    
#%% Beam hardening correction: 
import flexSpectrum

energy, spectrum = flexSpectrum.calibrate_spectrum(proj, vol,  meta['geometry'], compound = 'Al', density = 2.7, n_bin = 20, iterations = 10000)   

#%% Check:
proj_ = flexSpectrum.equivalent_density(proj,  meta['geometry'], energy, spectrum, compound = 'Al', density = 2.7) 

vol = flexProject.init_volume(proj)
flexProject.FDK(proj_, vol, meta['geometry'])

vol /= meta['geometry']['img_pixel'] ** 4

flexUtil.display_slice(vol, title = 'Corrected FDK')

# Save:
numpy.savetxt('/export/scratch2/kostenko/archive/OwnProjects/al_tests/new/90KV_1mm_brass/spectrum.txt', [energy, spectrum]) 
    
#energy, spec = numpy.loadtxt('/export/scratch2/kostenko/archive/OwnProjects/al_tests/new/90KV_1mm_brass/spectrum.txt')     
    
    