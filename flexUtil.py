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

def equivalent_density( energy, spectrum, compound, density):
    '''
    Transfer intensity values to equivalent density
    '''
    prnt = self._parent
    
    # Assuming that we have log data!
    #if not 'process.log(air_intensity, bounds)' in self._parent.meta.history.keys:                        
    #    self._parent.error('Logarithm was not found in history of the projection stack. Apply log first!')
    
    print('Generating the transfer function.')
    
    # Attenuation of 1 mm:
    mu = simulate.spectra.linear_attenuation(energy, compound, density, thickness = 0.1)
    width = self._parent.data.slice_shape[1]

    # Make thickness range that is sufficient for interpolation:
    thickness_min = 0
    thickness_max = width * self._parent.meta.geometry.img_pixel[1]
    
    print('Assuming thickness range:', [thickness_min, thickness_max])
    thickness = numpy.linspace(thickness_min, thickness_max, 1000)
    
    exp_matrix = numpy.exp(-numpy.outer(thickness, mu))
    synth_counts = exp_matrix.dot(spectrum)
    
    synth_counts = -numpy.log(synth_counts)
    
    plt.figure()
    plt.plot(thickness, synth_counts, 'r-', lw=4, alpha=.8)
    plt.axis('tight')
    plt.title('THickness (mm) v.s. absorption length.')
    plt.show()
    
    print('Callibration attenuation range:', [synth_counts[0], synth_counts[-1]])
    print('Data attenuation range:', [self._parent.analyse.min(), self._parent.analyse.max()])

    print('Applying transfer function.')    
    
    for ii, block in enumerate(self._parent.data):        
        block = numpy.array(numpy.interp(block, synth_counts, thickness * density), dtype = 'float32')
        
        prnt.data[ii] = block
        
        misc.progress_bar((ii+1) / self._parent.data.block_number)
        
    self._parent.meta.history.add_record('process.equivalent_thickness(energy, spectrum, compound, density)', [energy, spectrum, compound, density])    

def plot(x, y):
    
    plt.figure()
    plt.plot(x, y)

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