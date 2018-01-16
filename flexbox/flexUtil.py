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

def anyslice(array, index, dim):
    """
    Slice an array along an arbitrary dimension.
    """
    sl = [slice(None)] * array.ndim
    sl[dim] = index
      
    return sl

def cast2type(array, dtype, bounds = None):
    """
    Cast from float to int or float to float rescaling values if needed.
    """
    # No? Yes? OK...
    if array.dtype == dtype:
        return array
    
    # Make sue dtype is not a string:
    dtype = numpy.dtype(dtype)
    
    # If cast to float, simply cast:
    if dtype.kind == 'f':
        return numpy.array(array, dtype)
    
    # If to integer, rescale:
    if bounds is None:
        bounds = [numpy.amin(array), numpy.amax(array)]
    
    data_max = numpy.iinfo(dtype).max
    
    array -= bounds[0]
    array *= data_max / (bounds[1] - bounds[0])
    
    array[array < 0] = 0
    array[array > data_max] = data_max
    
    array = numpy.array(array, dtype)    
    
    return array

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

def plot(x, y = None, semilogy = False, title = None):
    
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
        
    plt.show()    

def display_slice(data, index = None, dim = 0, bounds = None, title = None):
    
    # If the image is 2D:
    if data.ndim == 2:
        plt.figure()
        plt.imshow(data)
        plt.colorbar()
        plt.show()
        return
        
    # Else:
        
    if index is None:
        index = data.shape[dim] // 2

    sl = anyslice(data, index, dim)

    img = numpy.squeeze(data[sl])
    
    plt.figure()
    if bounds:
        plt.imshow(img, vmin = bounds[0], vmax = bounds[1])
    else:
        plt.imshow(img)
        
    plt.colorbar()
    
    if title:
        plt.title(title)

def display_projection(data, dim = 0, title = None):
    
    img = data.sum(dim)
    
    plt.imshow(img)
    plt.colorbar()
    
    if title:
        plt.title(title)
        
def display_max_projection(data, dim = 0, title = None):
    
    img = data.max(dim)
    
    plt.imshow(img)
    plt.colorbar()
    
    if title:
        plt.title(title)     

def display_min_projection(data, dim = 0, title = None):
    
    img = data.min(dim)
    
    plt.imshow(img)
    plt.colorbar()
    
    if title:
        plt.title(title)         
