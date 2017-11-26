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

#%% Create volume and forward project:
    
# Initialize images:    
vol = numpy.zeros([1, 512, 512], dtype = 'float32')
proj = numpy.zeros([1, 361, 512], dtype = 'float32')

# Define a simple projection geometry:
src2obj = 100
det2obj = 100    
det_pixel = 0.1

geometry = flexData.create_geometry(src2obj, det2obj, det_pixel, [0, 360], 361)

# Create phantom and project into proj:
vol = flexModel.phantom(vol.shape, 'bubble', [150, 15])     
flexProject.forwardproject(proj, vol, geometry)

#%%
# Get the material refraction index:
c = flexSpectrum.find_nist_name('Calcium Carbonate')    
rho = c['density'] 
z = (numpy.array(c['Elements']) * numpy.array(c['massFractions'])).sum()

n_calcium = flexSpectrum.material_refraction('Calcium Carbonate', rho, z, energy)
proj = numpy.exp(-proj * numpy.imag(n_calcium))

# Display:
flexUtil.display_slice(proj) 
   
#%% Phase contrast effect:
    
energy = 30

# Ratio between phase and attenuation effects needed to construct propagator 
alpha = numpy.imag(n_calcium) / numpy.real(n_calcium)

# Propagator (Dual CTF):
p = flexModel.get_PSF(shape, mode = 'gaussian', [det_pixel, energy, src2obj, det2obj, alpha])

for ii in range(proj.shape[1]):
    img = proj[ii]
    img = flexModel.apply_PSF(img, p)

flexModel.apply_noise(proj, 1e4)

flexUtil.display_slice(proj)
