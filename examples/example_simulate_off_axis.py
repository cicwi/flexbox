#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test forward / backward projection of a 2D phantom.
"""
#%%
import flexbox as flex

import numpy

#%% Create volume and forward project:
    
# Initialize images:    
vol = numpy.zeros([1, 512, 512], dtype = 'float32')
proj = numpy.zeros([1, 361, 512], dtype = 'float32')

# Define a simple projection geometry:
geometry = flex.data.create_geometry(src2obj = 100, det2obj = 100, det_pixel = 0.01, theta_range = [0, 360], theta_count = 361)

# Create phantom and project into proj:
vol = flex.model.phantom(vol.shape, 'ball', [150, 15])     
flex.util.display_slice(vol)

geometry['axs_hrz'] = 80 * 0.01

# Forward project:
flex.project.forwardproject(proj, vol, geometry)
flex.util.display_slice(proj)

#%% Reconstruct

vol_rec = numpy.zeros_like(vol)

flex.project.FDK(proj, vol_rec, geometry)
flex.util.display_slice(vol_rec)

#%% Apply weighted FDK:

def ramp(data, length):
    
    data_ = data.copy()
    ramp = numpy.linspace(0, numpy.pi/2, length)
    ramp = numpy.sin(ramp)
    
    data_[:, :, :length] *= ramp[None, None, :]
    data_[:, :, -length:] *= ramp[None, None, ::-1]

    return data_

vol_rec = numpy.zeros_like(vol)    
vol_weight = numpy.zeros_like(vol)

#flex.project.FDK(proj * 0 + 1., vol_weight, geometry)
#flex.project.FDK(proj, vol_rec, geometry)

flex.project.FDK(ramp(proj * 0 + 1., 20), vol_weight, geometry)
flex.project.FDK(ramp(proj, 20), vol_rec, geometry)

vol_rec = vol_rec / (vol_weight**2 + 1e-5) * vol_rec


flex.util.display_slice(vol_rec)    
