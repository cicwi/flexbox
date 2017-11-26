#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test flexData module.
"""
#%%
import flexData
import flexProject
import flexUtil
import flexModel
import flexSpectrum

import numpy

#%% Define geometry:
    
# Initialize images:    
vol = numpy.zeros([1, 512, 512], dtype = 'float32')
proj = numpy.zeros([1, 361, 512], dtype = 'float32')

# Define a simple projection geometry:
src2obj = 100
der2obj = 100    
det_pixel = 0.1

geometry = flexData.create_geometry(src2obj, det2obj, det_pixel, [0, 360], 361)
energy = numpy.linspace(10, 90, 9)

spectrum = flexSpectrum.bremsstrahlung(energy, 90) 
spectrum *= flexSpectrum.scintillator_efficiency(energy, 'Si', rho = 5, 0.1)

flexUtil.plot(energy, spectrum)

#%% Model data:

# Create phantom and project into proj:
vol = flexModel.phantom(vol.shape, 'bubble', [150, 15])     
flexProject.forwardproject(proj, vol, geometry)

# Simulate intensity images:
counts = numpy.zeros_like(proj)

for ii in range(len(energy)):
    
    # Simulate phase contrast:
    counts += energy[ii] * spectrum[ii] * flexModel.simulate_phase(proj, 'Calcium Carbonate', energy[ii], geometry)
    
# Add noise and blurring:
counts = flexModel.apply_PSF(counts, 'gaussian', 0.1)
counts = flexModel.apply_noise(counts, 1)

# Display:
flexUtil.display_slice(counts) 
   