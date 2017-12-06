#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 2017
@author: kostenko

Displaying and stuff...
"""

''' * Imports * '''

import numpy
import matplotlib.pyplot as plt

''' * Methods * '''

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
    
    if index is None:
        index = data.shape[dim] // 2

    if dim == 0:
        img = data[index, :, :]

    elif dim == 1:
        img = data[:, index, :]

    elif dim == 2:
        img = data[:, :, index]

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