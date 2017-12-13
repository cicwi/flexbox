#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 2017
@author: kostenko

This module contains calculation routines for pre/post processing.
"""
import numpy
from scipy import ndimage

from . import flexUtil
from . import flexData
from . import flexProject
from . import misc

def rotate(data, angle, axis = 0):
    '''
    Rotates the volume via interpolation.
    '''
    
    print('Applying rotation.')
    
    misc.progress_bar(0)  
    
    sz = data.shape[axis]
    
    for ii in range(sz):     
        
        sl = misc.anyslice(data, ii, axis)
        
        data[sl] = ndimage.interpolation.rotate(data[sl], angle, reshape=False)
        
        misc.progress_bar((ii+1) / sz)
        
    return data
        
def translate(data, shift, axis = 0):
    """
    Apply a 2D tranlation perpendicular to the axis.
    """
    
    print('Applying translation.')
    
    misc.progress_bar(0)  
    
    sz = data.shape[axis]
    
    for ii in range(sz):     
        
        sl = misc.anyslice(data, ii, axis)
        
        data[sl] = ndimage.interpolation.shift(data[sl], shift, order = 1, reshape=False)
        
        misc.progress_bar((ii+1) / sz)   

    return data
    
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
        
    def interpolate_holes(self, mask2d, kernel = [3,3,3]):
        '''
        Fill in the holes, for instance, saturated pixels.
        
        Args:
            mask2d: holes are zeros. Mask is the same for all projections.
        '''
        
        misc.progress_bar(0)        
        for ii, block in enumerate(self._parent.data):    
                    
            # Compute the filler:
            tmp = ndimage.filters.gaussian_filter(mask2d, sigma = kernel)        
            tmp[tmp > 0] **= -1

            # Apply filler:                 
            block = block * mask2d[:, None, :]           
            block += ndimage.filters.gaussian_filter(block, sigma = kernel) * (~mask2d[:, None, :])
            
            self._parent.data[ii] = block   

            # Show progress:
            misc.progress_bar((ii+1) / self._parent.data.block_number)
            
        self._parent.meta.history.add_record('process.interpolate_holes(mask2d, kernel)', kernel)

def residual_rings(data, kernel=[3, 1, 3]):
    '''
    Apply correction by computing outlayers .
    '''
    import ndimage
    
    # Compute mean image of intensity variations that are < 5x5 pixels
    print('Our best agents are working on the case of the Residual Rings. This can take years if the kernel size is too big!')

    misc.progress_bar(0)        
    
    tmp = numpy.zeros(data.shape[::2])
    
    for ii in range(data.shape[1]):                 
        
        block = data[:, ii, :]

        # Compute:
        tmp += (block - ndimage.filters.median_filter(block, size = kernel)).sum(1)
        
        misc.progress_bar((ii+1) / data.shape[1])
        
    tmp /= data.shape[1]
    
    print('Subtract residual rings.')
    
    misc.progress_bar(0)        
    
    for ii in range(data.shape[1]):                 
        
        block = data[:, ii, :]
        block -= tmp

        misc.progress_bar((ii+1) / data.shape[1])
        
        data[:, ii, :] = block 
    
    print('Residual ring correcion applied.')
    return data

def subtract_air(data, air_val = None):
    '''
    Subtracts a coeffificient from each projection, that equals to the intensity of air.
    We are assuming that air will produce highest peak on the histogram.
    '''
    print('Air intensity will be derived from 10 pixel wide border.')
    
    # Compute air if needed:
    if air_val is None:  
        
        air_val = -numpy.inf
        
        for ii in range(data.shape[1]): 
            # Take pixels that belong to the 5 pixel-wide margin.
            
            block = data[:, ii, :]

            border = numpy.concatenate((block[:10, :].ravel(), block[-10:, :].ravel(), block[:, -10:].ravel(), block[:, :10].ravel()))
          
            y, x = numpy.histogram(border, 1024, range = [-0.1, 0.1])
            x = (x[0:-1] + x[1:]) / 2
    
            # Subtract maximum argument:    
            air_val = numpy.max([air_val, x[y.argmax()]])
    
    print('Subtracting %f' % air_val)  
    
    misc.progress_bar(0)  
    
    for ii in range(data.shape[1]):  
        
        block = data[:, ii, :]

        block = block - air_val
        block[block < 0] = 0
        
        data[:, ii, :] = block

        misc.progress_bar((ii+1) / data.shape[1])
        
    return data
                    
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



