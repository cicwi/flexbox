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

def print_memory():
    import psutil
    
    print('Free memory: %u GB (%u%% left)' % (psutil.virtual_memory().available/1e9, psutil.virtual_memory().available / psutil.virtual_memory().total * 100))
    
def add_dim(array_1, array_2):
    """
    Add two arrays with arbitrary dimensions. We assume that one or two dimensions match.
    """
    
    # Shapes to compare:
    shp1 = numpy.shape(array_1)
    shp2 = numpy.shape(array_2)
    
    dim1 = numpy.ndim(array_1)
    dim2 = numpy.ndim(array_2)
    
    if dim1 - dim2 == 1:
        
        # Find dimension that is missing in array_2:
        dim = [ii not in shp2 for ii in shp1].index(True)
        
        if dim == 0:
            array_1 += array_2[None, :, :]
        elif dim == 1:
            array_1 += array_2[:, None, :]
        elif dim == 2:
            array_1 += array_2[:, :, None]            
        
    elif dim1 - dim2 == 2:
        # Find dimension that is matching in array_2:
        dim = [ii in shp2 for ii in shp1].index(True)
        
        if dim == 0:
            array_1 += array_2[:, None, None]
        elif dim == 1:
            array_1 += array_2[None, :, None]
        else:
            array_1 += array_2[None, None, :]
            
    else:
        raise('ERROR! array_1.ndim - array_2.ndim should be 1 or 2')
            
    
def mult_dim(array_1, array_2):    
    """
    Multiply a 3D array by a 1D or a 2D vector along one of the dimensions.
    
    """
    # Shapes to compare:
    shp1 = numpy.shape(array_1)
    shp2 = numpy.shape(array_2)
    
    dim1 = numpy.ndim(array_1)
    dim2 = numpy.ndim(array_2)
    
    if dim1 - dim2 == 1:
        
        # Find dimension that is missing in array_2:
        dim = [ii not in shp2 for ii in shp1].index(True)
        
        if dim == 0:
            array_1 *= array_2[None, :, :]
        elif dim == 1:
            array_1 *= array_2[:, None, :]
        elif dim == 2:
            array_1 *= array_2[:, :, None]            
        
    elif dim1 - dim2 == 2:
        # Find dimension that is matching in array_2:
        dim = [ii in shp2 for ii in shp1].index(True)
        
        if dim == 0:
            array_1 *= array_2[:, None, None]
        elif dim == 1:
            array_1 *= array_2[None, :, None]
        else:
            array_1 *= array_2[None, None, :]
            
    else:
        raise('ERROR! array_1.ndim - array_2.ndim should be 1 or 2')
        
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
    
    # TODO: replace with tqdm!!!
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
        x = numpy.arange(numpy.size(x))
    
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
        
        # There is a bug in plt. It doesn't like float16
        if img.dtype == 'float16': img = numpy.float32(img)
        
    plt.figure()
    if bounds:
        plt.imshow(img, vmin = bounds[0], vmax = bounds[1], cmap = cmap)
    else:
        plt.imshow(img, cmap = cmap)
        
    plt.colorbar()
    
    if title:
        plt.title(title)
        
    plt.show()  
    
def display_mesh(stl_mesh):
    """
    Display an stl mesh. Use flexCompute.generate_stl(volume) to generate mesh.
    """    
    from mpl_toolkits import mplot3d
        
    figure = plt.figure()
    axes = mplot3d.Axes3D(figure)

    axes.add_collection3d(mplot3d.art3d.Poly3DCollection(stl_mesh.vectors))
    # Auto scale to the mesh size
    scale = stl_mesh.points.flatten(-1)
    axes.auto_scale_xyz(scale, scale, scale)
    # Show the plot to the screen
    plt.show()


def display_projection(data, dim = 1, bounds = None, title = None, cmap = 'gray'):
    
    img = data.sum(dim)
    
    # There is a bug in plt. It doesn't like float16
    img = numpy.float32(img)
    
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
    
    # There is a bug in plt. It doesn't like float16
    img = numpy.float32(img)
    
    plt.figure()
    
    plt.imshow(img, cmap = cmap)
    plt.colorbar()
    
    if title:
        plt.title(title)     
        
    plt.show()
        
def display_min_projection(data, dim = 0, title = None, cmap = 'gray'):
    
    img = data.min(dim)
    
    # There is a bug in plt. It doesn't like float16
    img = numpy.float32(img)
    
    plt.figure()
    
    plt.imshow(img, cmap = cmap)
    plt.colorbar()
    
    if title:
        plt.title(title)         
        
    plt.show()