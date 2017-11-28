#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test simulation including phase-contrast effect.
We will simulate conditions close to micro-CT of a sea shell.
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
vol = flexModel.phantom(vol.shape, 'bubble', [150, 15])     
flexProject.forwardproject(proj, vol, geometry)

#%%
# Get the material refraction index:
c = flexSpectrum.find_nist_name('Calcium Carbonate')    
rho = c['density']

energy = 50
n = flexSpectrum.material_refraction('CaCO3', rho, energy)

# This is instensity image:
#proj = numpy.exp(-proj * numpy.real(n * 2))

# Display:
flexUtil.display_slice(proj) 

#%% Proper Fresnel propagation for phase-contrast:
ctf = flexModel.get_ctf(proj.shape[::2], 'fresnel', [det_pixel, energy, src2obj, det2obj])
ctf_ = flexModel.get_ctf(proj.shape[::2], 'gaussian', [det_pixel, det_pixel])

# Electro-magnetic field image:
proj_em = numpy.exp(-proj * n)

# Intensity:
proj_i0 = numpy.abs(flexModel.apply_ctf(proj_em, ctf)) ** 2
proj_i0 = numpy.real(flexModel.apply_ctf(proj_i0, ctf_))
 
#flexUtil.display_slice(numpy.abs(proj_em)**2)                                              
flexUtil.display_slice(proj_i0)                           
#%% Phase contrast effect:
    
# Ratio between phase and attenuation effects needed to construct propagator 

# Propagator (Dual CTF):
#alpha = numpy.imag(n) / numpy.real(n)
#p = flexModel.get_PSF(proj.shape[::2], 'dual_ctf', [det_pixel, energy, src2obj, det2obj, alpha])

#for ii in range(proj.shape[1]):
#    img = proj[ii]
#    img = flexModel.apply_PSF(img, p)

#flexModel.apply_noise(proj, 1e4)

#flexUtil.display_slice(proj)
flexUtil.plot(numpy.real(ctf))
