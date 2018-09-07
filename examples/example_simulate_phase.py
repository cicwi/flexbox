#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test simulation including phase-contrast effect.
We will simulate conditions close to micro-CT of a sea shell.
"""
#%%
import flexbox as flex
import numpy

#%% Create volume and forward project:
    
# Initialize images:    
x = 5
h = 512 * x
vol = numpy.zeros([1, h, h], dtype = 'float32')
proj = numpy.zeros([1, 361, h], dtype = 'float32')

# Define a simple projection geometry:
src2obj = 100     # mm
det2obj = 100     # mm   
det_pixel = 0.001 / x # mm (1 micron)

geometry = flex.data.create_geometry(src2obj, det2obj, det_pixel, [0, 360])

# Create phantom (150 micron wide, 15 micron wall thickness):
vol = flex.model.phantom(vol.shape, 'bubble', [150*x, 30*x,1])     
vol += flex.model.phantom(vol.shape, 'bubble', [10*x, 3*x,1])     
flex.project.forwardproject(proj, vol, geometry)

#%%
# Get the material refraction index:
c = flex.spectrum.find_nist_name('Calcium Carbonate')    
rho = c['density'] / 10

energy = 30 # KeV
n = flex.spectrum.material_refraction(energy, 'CaCO3', rho)

#%% Proper Fresnel propagation for phase-contrast:
   
# Create Contrast Transfer Functions for phase contrast effect and detector blurring    
phase_ctf = flex.model.get_ctf(proj.shape[::2], 'fresnel', [det_pixel, energy, src2obj, det2obj])

sigma = det_pixel 
phase_ctf *= flex.model.get_ctf(proj.shape[::2], 'gaussian', [det_pixel, sigma * 1])

# Electro-magnetic field image:
proj_i = numpy.exp(-proj * n )


# Field intensity:
proj_i = flex.model.apply_ctf(proj_i, phase_ctf) ** 2

#proj_i = numpy.abs(numpy.exp(-proj * n ))**2

flex.util.display_slice(proj_i, title = 'Projections (phase contrast)')

#%% Reconstruct with phase contrast:
    
vol_rec = numpy.zeros_like(vol)

flex.project.FDK(-numpy.log(proj_i), vol_rec, geometry)
flex.util.display_slice(vol_rec, title = 'FDK')  
    
#%% Invertion of phase contrast based on dual-CTF model:
    
# Propagator (Dual CTF):
alpha = numpy.imag(n) / numpy.real(n)
dual_ctf = flex.model.get_ctf(proj.shape[::2], 'dual_ctf', [det_pixel, energy, src2obj, det2obj, alpha])
dual_ctf *= flex.model.get_ctf(proj.shape[::2], 'gaussian', [det_pixel, sigma])

# Use inverse convolution to solve for blurring and pci
proj_inv = flex.model.deapply_ctf(proj_i, dual_ctf, epsilon = 0.1)

# Depending on epsilon there is some lof frequency bias introduced...
proj_inv /= proj_inv.max()

flex.util.display_slice(proj_inv, title = 'Inverted phase contrast')   

# Reconstruct:
vol_rec = numpy.zeros_like(vol)
flex.project.FDK(-numpy.log(proj_inv), vol_rec, geometry)
flex.util.display_slice(vol_rec, title = 'FDK')   

#%% SIRT algebraic deconvolution:
 
vol_rec = numpy.zeros_like(vol)    
options = {'bounds':[0, 10], 'l2_update':True, 'block_number':2, 'index':'sequential', 'ctf':dual_ctf}
flex.project.SIRT(-numpy.log(proj_i), vol_rec, geometry, iterations = 10, options = options)

flex.util.display_slice(vol_rec, title = 'SIRT')
