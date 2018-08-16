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
geometry = flex.data.create_geometry(src2obj = 100, det2obj = 100, det_pixel = 0.01, theta_range = [0, 360], type = 'simple')

# Create phantom and project into proj:
vol = flex.model.phantom(vol.shape, 'bubble', [150, 15, 1.5])     
vol = flex.compute.rotate(vol, 10, 0)

flex.util.display_slice(vol)

# Forward project:
flex.project.forwardproject(proj, vol, geometry)
flex.util.display_slice(proj)

#%% Reconstruct

vol_rec = numpy.zeros_like(vol)

flex.project.FDK(proj, vol_rec, geometry)
flex.util.display_slice(vol_rec)

#%% EM
vol_rec = numpy.zeros_like(vol)

options = {'bounds':[0, 10], 'l2_update':True, 'block_number':5, 'mode':'random'}
flex.project.EM(proj, vol_rec, geometry, iterations = 10, options = options)
flex.util.display_slice(vol_rec)

#%% SIRT
vol = numpy.zeros([1, 512, 512], dtype = 'float32')

options = {'bounds':[0, 10], 'l2_update':True, 'block_number':5, 'mode':'random'}
flex.project.SIRT(proj, vol, geometry, iterations = 10, options = options)

flex.util.display_slice(vol, title = 'SIRT')
