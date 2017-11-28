#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulate spectral data
"""
#%%
import flexData
import flexProject
import flexUtil
import flexModel
import flexSpectrum

import numpy

#%% Create volume and forward project:
    
# Initialize images:    
vol = numpy.zeros([1, 512, 512], dtype = 'float32')
proj = numpy.zeros([1, 361, 512], dtype = 'float32')

# Define a simple projection geometry:
src2obj = 100   # mm
det2obj = 100   # mm   
det_pixel = 0.001 # mm (1 micron)

geometry = flexData.create_geometry(src2obj, det2obj, det_pixel, [0, 360], 361)

# Create phantom (150 micron wide, 15 micron wall thickness):
vol = flexModel.phantom(vol.shape, 'pearl', [150,])     
flexProject.forwardproject(proj, vol, geometry)

#%% Simulate spectrum:

energy = numpy.linspace(10, 90, 9)

spectrum = flexSpectrum.bremsstrahlung(energy, 90) 
spectrum *= flexSpectrum.total_transmission(energy, 'Cu', 8, 0.1)
spectrum *= flexSpectrum.scintillator_efficiency(energy, 'Si', rho = 5, thickness = 0.1)
    
spectrum /= (energy*spectrum).sum()

# Get the material refraction index:
c = flexSpectrum.find_nist_name('Calcium Carbonate')    
rho = c['density'] * 5
n = flexSpectrum.material_refraction('CaCO3', rho, energy)
 
# Display:
flexUtil.plot(energy, spectrum) 

#%% Model data:
    
# Create phantom and project into proj:
#vol = flexModel.phantom(vol.shape, 'bubble', [150, 15])     
#flexProject.forwardproject(proj, vol, geometry)

# Simulate intensity images:
counts = numpy.zeros_like(proj)

for ii in range(len(energy)):
    
    # Simulate phase contrast:
    counts += energy[ii] * spectrum[ii] * numpy.exp(-proj * numpy.real(n[ii] * 2))
    
#counts = flexModel.apply_noise(counts, 1)

# Display:
flexUtil.display_slice(counts) 

#%% Reconstruct:
    
vol_rec = numpy.zeros_like(vol)

flexProject.FDK(-numpy.log(counts), vol_rec, geometry)
flexUtil.display_slice(vol_rec)
    
#%% Beam hardening correction: 
vol_rec *= 0
flexProject.FDK(proj, vol_rec, geometry)
flexUtil.display_slice(vol_rec)   