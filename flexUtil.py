#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 2017
@author: kostenko

Displaying and stuff...
"""

''' * Imports * '''

import numpy
import misc
import matplotlib.pyplot as plt

''' * Methods * '''

def plot(x, y = None, title = None):
    
    if y is None:
        y = x
        x = numpy.arange(x.size)
    
    x = numpy.squeeze(x)
    y = numpy.squeeze(y)
    
    plt.figure()
    plt.plot(x, y)
    
    if title:
        plt.title(title)

def display_slice(data, index = None, dim = 0, title = None):
    
    if index is None:
        index = data.shape[dim] // 2

    if dim == 0:
        img = data[index, :, :]

    elif dim == 1:
        img = data[:, index, :]

    elif dim == 2:
        img = data[:, :, index]

    plt.figure()
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