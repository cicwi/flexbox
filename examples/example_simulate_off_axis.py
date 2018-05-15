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
geometry = flex.data.create_geometry(src2obj = 100, det2obj = 100, det_pixel = 0.01, theta_range = [0, 360])

# Create phantom and project into proj:
vol = flex.model.phantom(vol.shape, 'ball', [150, 1])     
flex.util.display_slice(vol, title = 'phantom')

geometry['axs_hrz'] = 1

# Forward project:
flex.project.forwardproject(proj, vol, geometry)
flex.util.display_slice(proj)

vol_rec = numpy.zeros_like(vol)
vol_rec *= 0

flex.project.FDK(proj, vol_rec, geometry)
flex.util.display_slice(vol_rec)

#%% Apply ramp:
vol_rec *= 0 

flex.project.FDK(flex.util.apply_edge_ramp(proj, [0, 200]), vol_rec, geometry)

flex.util.display_slice(vol_rec)    
