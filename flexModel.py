#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 2017

@author: kostenko

This module includes a few routines useful for modeling polychromatic X-ray CT data

"""
import numpy

def phantom(shape, mode = 'bubble', parameters = [10, 1]):
    """
    Create a phantom image.
    
    Args:
        shape (list): shape of the volume 
        type (str): use 'bubble' or 'pearl'
        parameters (list or float): for the bubble - [outer radius, wall thickness], pearl - radius
    """    
    
    xx = numpy.arange(0, shape[0]) - shape[0] // 2
    yy = numpy.arange(0, shape[1]) - shape[1] // 2
    zz = numpy.arange(0, shape[2]) - shape[2] // 2
                                  
    vol = (xx[:, None, None]**2 + yy[None, :, None]**2 + zz[None, None, :]**2)

    if mode == 'bubble':
        r0 = (parameters[0] - parameters[1])**2
        r1 = (parameters[0])**2
    
        vol = numpy.array(((vol > r0) & (vol < r1)), dtype = 'float32')                 
    elif mode == 'pearl':
        r0 = parameters[0] ** 2
        vol = numpy.array((vol > r0), dtype = 'float32') 

    return vol                

def get_PSF(shape, mode = 'gaussian', parameter = 1):
    """
    Get a PSF of one of the following types: gaussian, dual_ctf
    
    Args:
        shape (list): shape of a projection image
        mode (str): 'gaussian', 'dual_ctf' (phase contrast)
        parameter (list / float): psf parameters. For dual_ctf: [detector_pixel, energy, src2obj, det2obj, alpha]  
        
    Returns:
        
    """
    if mode == 'gaussian':
        
        # Gaussian PSF:
        sigma = parameter
        center = shape // 2
          
        xx = numpy.arange(0, shape[0])[:, None]
        yy = numpy.arange(0, shape[0])[None, :]
             
        return 1 / (2 * numpy.pi * sigma**2) * numpy.exp(-((xx / sigma)**2 + (yy / sigma)**2) / 2)
    
    elif mode == 'dual_ctf':
        
        # Dual CTF propagator:
        pixelsize = parameter[0]
        energy = parameter[1]
        r1 = parameter[2]
        r2 = parameter[3]
        alpha = parameter[4]
        
        frequency = _frequency_space_(shape, pixelsize, energy, r1, r2)  
        
        return -2 * numpy.cos(frequency) + 2 * (alpha) * numpy.sin(frequency)
                          
def _frequency_space_(shape, pixelsize, energy, src2obj, det2obj):
    """
    Generates the lambda*freq**2*R image that can be used to calculate phase propagator at distance R, photon wavelength lambda.
    """
    # Image dimentions:
    sz = shape * pixelsize

    # Wavenumber;
    k = energy / (flexSpectrum.phys_const['h_bar_ev'] * flexSpectrum.phys_const['c'])
    
    # Frequency squared:
    u = 2 * numpy.pi * numpy.arange(shape[0]) / sz[0]
    v = 2 * numpy.pi * numpy.arange(shape[1]) / sz[1]
    w2 = numpy.fft.fftshift(u**2[:, None] + v**2[None, :])
         
    # W**2 * R:
    m = (src2obj + det2obj) / src2obj
    r_eff = det2obj / m
    
    return w2 * r_eff / (2 * k)
          
def apply_PSF(image, psf):
    """
    Apply psf to the image using convolution.
    """
    return scipy.signal.fftconvolve(image, psf)
      
def apply_phase_contrast():
    pass

def apply_noise(image, mode = 'poisson', parameter = 1):
    """
    Add noise to the data.
    
    Args:
        image (numpy.array): image to apply noise to
        mode (str): poisson or normal
        parameter (float): norm factor for poisson or a standard deviation    
    """
    
    if mode == 'poisson':
        image = numpy.random.poisson(image * parameter)
        
    elif mode == 'normal':
        image = numpy.random.normal(image, parameter)
        
    else: 
        raise ValueError('Me not recognize the mode! Use normal or poisson!')
