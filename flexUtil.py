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

def display_slice(data, index = None, dim = 0):
    
    if index is None:
        index = data.shape[dim] // 2

    if dim == 0:
        img = data[index, :, :]

    elif dim == 1:
        img = data[:, index, :]

    elif dim == 2:
        img = data[:, :, index]

    plt.imshow(img)
    plt.colorbar()

def display_projection(data, dim = 0):
    
    img = data.sum(dim)
    
    plt.imshow(img)
    plt.colorbar()