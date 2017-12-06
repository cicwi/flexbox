#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 2017
@author: kostenko

This module contains calculation routines for pre/post processing.
"""
import numpy
import flexUtil
import flexData
import flexProject

def histogram(data, nbin = 256, plot = True, log = False):
    """
    Compute histogram of the data.
    """
    
    print('Calculating histogram...')
    
    mi = data.min()
    ma = data.max()

    y, x = numpy.histogram(data, bins = nbin, range = [mi, ma])
        
    # Set bin values to the middle of the bin:
    x = (x[0:-1] + x[1:]) / 2

    flexUtil.plot(x, y, semilogy = True, title = 'Histogram')
    
    return x, y
    
def centre(data):
        """
        Compute the centre of the square of mass.
        """
        data2 = data.copy()**2
        
        M00 = data2.sum()
                
        return [moment(data2, 1, 0) / M00, moment(data2, 1, 1) / M00, moment(data2, 1, 2) / M00]
        
def moment(data, power, dim, centered = True):
    """
    Compute image moments (weighed averages) of the data. 
    
    sum( (x - x0) ** power * data ) 
    
    Args:
        power (float): power of the image moment
        dim (uint): dimension along which to compute the moment
        centered (bool): if centered, center of coordinates is in the middle of array.
        
    """
    # Create central indexes:
    shape = data.shape

    # Index:        
    x = numpy.arange(0, shape[dim])    
    if centered:
        x -= shape[dim] // 2
    
    x **= power
    
    if dim == 0:
        return numpy.sum(x[:, None, None] * data)
    elif dim == 1:
        return numpy.sum(x[None, :, None] * data)
    else:
        return numpy.sum(x[None, None, :] * data)
    
def _parabolic_min_(values, index, space):    
    '''
    Use parabolic interpolation to find the extremum close to the index value:
    '''
    if (index > 0) & (index < (values.size - 1)):
        # Compute parabolae:
        x = space[index-1:index+2]    
        y = values[index-1:index+2]

        denom = (x[0]-x[1]) * (x[0]-x[2]) * (x[1]-x[2])
        A = (x[2] * (y[1]-y[0]) + x[1] * (y[0]-y[2]) + x[0] * (y[2]-y[1])) / denom
        B = (x[2]*x[2] * (y[0]-y[1]) + x[1]*x[1] * (y[2]-y[0]) + x[0]*x[0] * (y[1]-y[2])) / denom
            
        x0 = -B / 2 / A  
        
    else:
        
        x0 = space[index]

    return x0    
    
def _modifier_l2cost_(projections, geometry, subsample, value, key = 'axs_hrz', display = False):
    '''
    Cost function based on L2 norm of the first derivative of the volume. Computation of the first derivative is done by FDK with pre-initialized reconstruction filter.
    '''
    geometry_ = geometry.copy()
    
    geometry_[key] = value

    vol = flexProject.sample_FDK(projections, geometry_, subsample)

    l2 = 0
    
    for ii in range(vol.shape[0]):
        grad = numpy.gradient(numpy.squeeze(vol[ii, :, :]))
        
        grad = (grad[0] ** 2 + grad[1] ** 2)         
        
        l2 += numpy.sum(grad)
        
    if display:
        flexUtil.display_slice(vol, title = 'Guess = %0.2e, L2 = %0.2e'% (value, l2))    
            
    return -l2    
    
def _optimize_modifier_subsample_(values, projections, geometry, samp = [1, 1], key = 'axs_hrz', display = True):  
    '''
    Optimize a modifier using a particular sampling of the projection data.
    '''  
    maxiter = values.size
    
    # Valuse of the objective function:
    func_values = numpy.zeros(maxiter)    
    
    print('Starting a full search for: ' , values)
    
    ii = 0
    for val in values:
        
        #print('Step %0d / %0d' % (ii+1, maxiter))
        
        func_values[ii] = _modifier_l2cost_(projections, geometry, samp, val, 'axs_hrz', display)

        ii += 1          
    
    min_index = func_values.argmin()    
    
    return _parabolic_min_(func_values, min_index, values)  
        
def optimize_rotation_center(projections, geometry, guess = None, subscale = 1, centre_of_mass = True):
    '''
    Find a center of rotation. If you can, use the center_of_mass option to get the initial guess.
    If that fails - use a subscale larger than the potential deviation from the center. Usually, 8 or 16 works fine!
    '''
    
    # Usually a good initial guess is the center of mass of the projection data:
    if  (centre_of_mass) & (guess is None):  
        
        print('Computing centre of mass...')
        
        guess = flexData.pixel2mm(centre(projections)[2])
        
    elif guess is None:
        
        guess = geometry('axs_tra')
        
    img_pix = geometry['det_pixel'] / ((geometry['src2obj'] + geometry['det2obj']) / geometry['src2obj'])
    
    print('The initial guess for the rotation axis shift is %0.2e mm' % guess)
    
    # Downscale the data:
    while subscale >= 1:
        
        # Check that subscale is 1 or divisible by 2:
        if (subscale != 1) & (subscale // 2 != subscale / 2): ValueError('Subscale factor should be a power of 2! Aborting...')
        
        print('Subscale factor %1d' % subscale)    

        # We will use constant subscale in the vertical direction but vary the horizontal subscale:
        samp =  [20, subscale]

        # Create a search space of 5 values around the initial guess:
        trial_values = numpy.linspace(guess - img_pix * subscale, guess + img_pix * subscale, 5)
        
        guess = _optimize_modifier_subsample_(trial_values, projections, geometry, samp, key = 'axs_hrz', display = False)
                
        print('Current guess is %0.2e mm' % guess)
        
        subscale = subscale // 2
    
    return guess



