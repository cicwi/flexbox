#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulate spectral data with Poisson noise
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
src2obj = 100     # mm
det2obj = 100     # mm   
det_pixel = 0.2   # mm (100 micron)

geometry = flexData.create_geometry(src2obj, det2obj, det_pixel, [0, 360], 361)

# Create phantom (150 micron wide, 15 micron wall thickness):
vol = flexModel.phantom(vol.shape, 'ball', [150,])     
flexProject.forwardproject(proj, vol, geometry)

#%% Simulate spectrum:

energy = numpy.linspace(10, 80, 100)

# Tube:
spectrum = flexSpectrum.bremsstrahlung(energy, 90) 
spectrum[20] = 3
# Filter:
spectrum *= flexSpectrum.total_transmission(energy, 'Cu', 8, 0.1)
# Detector:
spectrum *= flexSpectrum.scintillator_efficiency(energy, 'Si', rho = 5, thickness = 1)
# Normalize:
spectrum /= (energy*spectrum).sum()

# Get the material refraction index:
mu = flexSpectrum.linear_attenuation(energy, 'Al', 2.7)
 
# Display:
flexUtil.plot(energy, spectrum, 'Spectrum') 
flexUtil.plot(energy, mu, 'Linear attenuation') 

#%% Model data:
    
# Simulate intensity images:
counts = numpy.zeros_like(proj)

n_phot = 1e7

for ii in range(len(energy)):
    
    # Monochromatic component:
    monochrome = spectrum[ii] * numpy.exp(-proj * mu[ii])
    monochrome = flexModel.apply_noise(monochrome, 'poisson', n_phot) / n_phot    
    
    # Detector response is assumed to be proportional to E
    counts += energy[ii] * monochrome

# Simulate detector blurring:
ctf = flexModel.get_ctf(counts.shape[::2], 'gaussian', [det_pixel, det_pixel])
counts = flexModel.apply_ctf(counts, ctf)        

# Display:
flexUtil.display_slice(counts, title = 'Modelled sinogram') 

#%% Reconstruct:
    
vol_rec = numpy.zeros_like(vol)
proj_0 = -numpy.log(counts)

flexProject.FDK(proj_0, vol_rec, geometry)
flexUtil.display_slice(vol_rec, title = 'Uncorrected FDK')
    
#%% Beam hardening correction: 
proj_0 = -numpy.log(counts)

energy, spectrum = flexSpectrum.calibrate_spectrum(proj_0, vol_rec, geometry, compound = 'Al', density = 2.7, n_bin = 50)   
proj_0 = flexSpectrum.equivalent_density(proj_0, geometry, energy, spectrum, compound = 'Al', density = 2.7) 

flexProject.FDK(proj_0, vol_rec, geometry)
flexUtil.display_slice(vol_rec, title = 'Corrected FDK')
    