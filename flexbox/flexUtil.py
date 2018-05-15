#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 2017
@author: kostenko

Displaying data and and other small useful routines.
"""

''' * Imports * '''

import time
import numpy
import matplotlib.pyplot as plt

''' * Methods * '''

# Use global time variable to measure time needed to compute stuff:        
glob_time = 0

def apply_edge_ramp(data, width, extend = True):
    '''
    Apply ramp to the fringe of the tile to reduce artefacts.
    '''
    if numpy.size(width)>1:
        w0 = width[0]
        w1 = width[1]

    else:   
        w0 = width
        w1 = width
    
    # Pad the data:
    if extend:
        data = numpy.pad(data, ((w0, w0), (0,0),(w1, w1)), mode = 'linear_ramp', end_values = 0)
        
    else:
        if data.shape[0] > width*2:
            data[-width:, :, :] *= numpy.linspace(1, 0, width)[:, None, None]
            data[:width, :, :] *= numpy.linspace(0, 1, width)[:, None, None]

        data[:, :, -width:] *= numpy.linspace(1, 0, width)[None, None, :]
        data[:, :, :width] *= numpy.linspace(0, 1, width)[None, None, :]
        
    return data

def mult_dim(array, vector, dim):
    """
    Multiply a 3D array by a 1D vector along one of the dimensions.
    """
    if dim == 0:
        array *= vector[:, None, None]
        
    elif dim == 1:
        array *= vector[None, :, None]
        
    else:
        array *= vector[None, None, :]

def anyslice(array, index, dim):
    """
    Slice an array along an arbitrary dimension.
    """
    sl = [slice(None)] * array.ndim
    sl[dim] = index
      
    return sl
    
def progress_bar(progress):
    """
    Plot progress in pseudographics:
    """
    global glob_time 
    
    
    if glob_time == 0:
        glob_time = time.time()
    
    print('\r', end = " ")
    
    bar_length = 40
    if progress >= 1:
        
        # Repoort on time:
        txt = 'Done in %u sec!' % (time.time() - glob_time)
        glob_time = 0
        
        for ii in range(bar_length):
            txt = txt + ' '
            
        print(txt) 

    else:
        # Build a progress bar:
        txt = '\u2595'
        
        for ii in range(bar_length):
            if (ii / bar_length) <= progress:
                txt = txt + '\u2588'
            else:
                txt = txt + '\u2592'
                
        txt = txt + '\u258F'        
        
        print(txt, end = " ") 

def plot(x, y = None, semilogy = False, title = None, legend = None):
    
    if y is None:
        y = x
        x = numpy.arange(x.size)
    
    x = numpy.squeeze(x)
    y = numpy.squeeze(y)
    
    plt.figure()
    if semilogy:
        plt.semilogy(x, y)
    else:
        plt.plot(x, y)
    
    if title:
        plt.title(title)
    
    if legend:
        plt.legend(legend)
        
    plt.show()    

def display_slice(data, index = None, dim = 0, bounds = None, title = None, cmap = 'gray'):
    
    # Just in case squeeze:
    data = numpy.squeeze(data)
    
    # If the image is 2D:
    if data.ndim == 2:
        img = data
        
    else:
        if index is None:
            index = data.shape[dim] // 2
    
        sl = anyslice(data, index, dim)
    
        img = numpy.squeeze(data[sl])
        
    plt.figure()
    if bounds:
        plt.imshow(img, vmin = bounds[0], vmax = bounds[1], cmap = cmap)
    else:
        plt.imshow(img, cmap = cmap)
        
    plt.colorbar()
    
    if title:
        plt.title(title)
        
    plt.show()    

def display_projection(data, dim = 1, bounds = None, title = None, cmap = 'gray'):
    
    img = data.sum(dim)
    
    plt.figure()
    
    if bounds:
        plt.imshow(img, vmin = bounds[0], vmax = bounds[1], cmap = cmap)
    else:
        plt.imshow(img, cmap = cmap)
        
    plt.colorbar()
    
    if title:
        plt.title(title)
    
    plt.show()
    
def display_max_projection(data, dim = 0, title = None, cmap = 'gray'):
    
    img = data.max(dim)
    
    plt.imshow(img, cmap = cmap)
    plt.colorbar()
    
    if title:
        plt.title(title)     
        
    plt.show()
        
def display_min_projection(data, dim = 0, title = None, cmap = 'gray'):
    
    img = data.min(dim)
    
    plt.imshow(img, cmap = cmap)
    plt.colorbar()
    
    if title:
        plt.title(title)         
        
    plt.show()