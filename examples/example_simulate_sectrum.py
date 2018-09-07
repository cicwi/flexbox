#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulate spectral data with Poisson noise
"""
#%%
import flexbox as flex
import numpy
   
#%% Define a simple projection geometry:
geometry = flex.data.create_geometry(src2obj = 100, det2obj = 100, det_pixel = 0.2, theta_range = [0, 360])

#%% Short version:
vol = flex.model.phantom([1, 128, 128], 'box', [25,25,35])   
    
# Spectrum:    
E, S = flex.model.effective_spectrum(kv = 90)  
flex.util.plot(E,S, title ='Spectrum')   
  
mats = [{'material':'Al', 'density':2.7},]
                
counts = numpy.zeros([1, 128, 128], dtype = 'float32')

# Simulate:
flex.model.forward_spectral(vol, counts, geometry, mats, E, S, n_phot = 1e6)

# Display:
flex.util.display_slice(counts, title = 'Modelled sinogram')  

#%% Reconstruct:
    
vol_rec = numpy.zeros_like(vol)
proj = -numpy.log(counts)

flex.project.FDK(proj, vol_rec, geometry)
flex.util.display_slice(vol_rec, title = 'Uncorrected FDK')
flex.util.plot(vol_rec[0, 64])    
       
#%% Beam hardening correction: 

proj = -numpy.log(counts)
energy, spectrum = flex.compute.calibrate_spectrum(proj, vol_rec, geometry, compound = 'Al', density = 2.7, n_bin = 50)   
proj = flex.compute.equivalent_density(proj, geometry, energy, spectrum, compound = 'Al', density = 2.7) 

flex.project.FDK(proj, vol_rec, geometry)
flex.util.display_slice(vol_rec, title = 'Corrected FDK')
flex.util.plot(vol_rec[0, 64])