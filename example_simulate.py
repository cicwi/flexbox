#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test forward / backward projection of a 2D phantom.
"""
#%%
import flexData
import flexProject
import flexUtil
import flexModel

import numpy

#%% Create volume and forward project:
    
# Initialize images:    
vol = numpy.zeros([1, 512, 512], dtype = 'float32')
proj = numpy.zeros([1, 361, 512], dtype = 'float32')

# Define a simple projection geometry:
geometry = flexData.create_geometry(src2obj = 100, det2obj = 100, det_pixel = 0.01, theta_range = [0, 360], theta_count = 361)

# Create phantom and project into proj:
vol = flexModel.phantom(vol.shape, 'bubble', [150, 15])     
flexUtil.display_slice(vol)

# Forward project:
flexProject.forwardproject(proj, vol, geometry)
flexUtil.display_slice(proj)

#%% Reconstruct

vol_rec = numpy.zeros_like(vol)

flexProject.FDK(proj, vol_rec, geometry)
flexUtil.display_slice(vol_rec)