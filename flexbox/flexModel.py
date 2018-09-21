#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 2017

@author: kostenko

This module includes a few routines useful for modeling polychromatic X-ray CT data

"""
import numpy
from . import flexSpectrum
from . import flexUtil
from . import flexProject

class phantom():
    """
    Genereation of CT phantoms from geometrical primitives. Reads geometry data
    to compute dimensions correctly.
    """
    
    @staticmethod
    def _coords_(shape, geometry, offset = [0.,0.,0.]):
        """
        Coordinate space in mm.
        """
        voxel = geometry['img_pixel']
        
        xx = (numpy.arange(0, shape[0]) - shape[0] / 2) * voxel - offset[0] 
        yy = (numpy.arange(0, shape[1]) - shape[1] / 2) * voxel - offset[1]
        zz = (numpy.arange(0, shape[2]) - shape[2] / 2) * voxel - offset[2]
        
        return xx, yy, zz
    
    @staticmethod
    def sphere(shape, geometry, r, offset = [0., 0., 0.]):
        """
        Make sphere. Radius is in units (geometry['unit'])
        """
        return phantom.spheroid(shape, geometry, r, r, r, offset)
        
    @staticmethod
    def spheroid(shape, geometry, r1, r2, r3, offset = [0., 0., 0.]):
        """
        Make a spheroid. 
        """
        # Get the coordinates in mm:
        xx,yy,zz = phantom._coords_(shape, geometry, offset)
        
        # Volume init: 
        return  numpy.array((((xx[:, None, None]/r1)**2 + (yy[None, :, None]/r2)**2 + (zz[None, None, :]/r3)**2) < 1), dtype = 'float32') 
        
    @staticmethod
    def cuboid(shape, geometry, a, b, c, offset = [0., 0., 0.]):
        """
        Make a cuboid. Dimensions are in units (geometry['unit'])
        """
        # Get the coordinates in mm:
        xx,yy,zz = phantom._coords_(shape, geometry, offset)
         
        return  numpy.array((abs(xx[:, None, None]) < a / 2) * (abs(yy[None, :, None]) < b / 2) * (abs(zz[None, None, :]) < c / 2), dtype = 'float32')  
        
    @staticmethod        
    def cylinder(shape, geometry, r, h, offset = [0., 0., 0.]):
        """
        Make a cylinder with a specified radius and height.
        """
        
        volume = numpy.zeros(shape, dtype = 'float32')
        
        # Get the coordinates in mm:
        xx,yy,zz = phantom._coords_(shape, geometry, offset)
         
        volume = numpy.array(((zz[None, None, :])**2 + (yy[None, :, None])**2) < r ** 2, dtype = 'float32')  
        
        return (numpy.abs(xx) < h / 2)[:, None, None] * volume

    @staticmethod        
    def checkers(shape, geometry, frequency, offset = [0., 0., 0.]):
        """
        Make a 3D checkers board.
        """
        
        volume = numpy.zeros(shape, dtype = 'float32')
        
        # Get the coordinates in mm:
        xx,yy,zz = phantom._coords_(shape, geometry, offset)
        
        volume_ = numpy.zeros(shape, dtype='bool')
        
        step = shape[1] // frequency
        
        for ii in range(0, frequency):
            sl = slice(ii*step, int((ii + 0.5) * step))
            volume_[sl, :, :] = ~volume_[sl, :, :]
        
        for ii in range(0, frequency):
            sl = slice(ii*step, int((ii + 0.5) * step))
            volume_[:, sl, :] = ~volume_[:, sl, :]

        for ii in range(0, frequency):
            sl = slice(ii*step, int((ii + 0.5) * step))
            volume_[:, :, sl] = ~volume_[:, :, sl]
 
        volume *= volume_
        
        return volume
                
def get_ctf(shape, mode = 'gaussian', parameter = 1):
    """
    Get a CTF (fft2(PSF)) of one of the following types: gaussian, dual_ctf, fresnel
    
    Args:
        shape (list): shape of a projection image
        mode (str): 'gaussian', 'dual_ctf' (phase contrast)
        parameter (list / float): psf parameters. 
                  For gaussian: [detector_pixel, sigma]
                  For dual_ctf: [detector_pixel, energy, src2obj, det2obj, alpha]  
        
    Returns:
        
    """
    if mode == 'gaussian':
        
        # Gaussian CTF:
        pixel = parameter[0]
        sigma = parameter[1]
          
        u = _w_space_(shape, 0, pixel)
        v = _w_space_(shape, 1, pixel)
        
        ctf = numpy.exp(-((u * sigma) ** 2 + (v * sigma) ** 2)/2)
        #ctf = 1 / (2 * numpy.pi * sigma**2) * numpy.exp(-((xx / sigma)**2 + (yy / sigma)**2) / 2)
                   
        return numpy.fft.fftshift(ctf) 
    
    elif mode == 'dual_ctf':
        
        # Dual CTF approximation phase contrast propagator:
        pixelsize = parameter[0]
        energy = parameter[1]
        r1 = parameter[2]
        r2 = parameter[3]
        alpha = parameter[4]
        
        # Effective propagation distance:
        m = (r1 + r2) / r1
        r_eff = r2 / m
        
        # Wavenumber;
        k = energy / (flexSpectrum.phys_const['h_bar_ev'] * flexSpectrum.phys_const['c'])
        
        # Frequency square:
        w2 = _w2_space_(shape, pixelsize)
        
        #return -2 * numpy.cos(w2 * r_eff / (2*k)) + 2 * (alpha) * numpy.sin(w2 * r_eff / (2*k))
        return numpy.cos(w2 * r_eff / (2*k)) - (alpha) * numpy.sin(w2 * r_eff / (2*k))
    
    elif mode == 'fresnel':
        
        # Fresnel propagator for phase contrast simulation:
        pixelsize = parameter[0]
        energy = parameter[1]
        r1 = parameter[2]
        r2 = parameter[3]
        
        # Effective propagation distance:
        m = (r1 + r2) / r1
        r_eff = r2 / m
        
        # Wavenumber;
        k = energy / (flexSpectrum.phys_const['h_bar_ev'] * flexSpectrum.phys_const['c'])
        
        # Frequency square:
        w2 = _w2_space_(shape, pixelsize)
        
        return numpy.exp(1j * w2 * r_eff / (2*k))
        
    elif mode == 'tie':
        
        # Transport of intensity equation approximation of phase contrast:
        pixelsize = parameter[0]
        energy = parameter[1]
        r1 = parameter[2]
        r2 = parameter[3]
        alpha = parameter[4]
        
        # Effective propagation distance:
        m = (r1 + r2) / r1
        r_eff = r2 / m
        
        # Wavenumber;
        k = energy / (flexSpectrum.phys_const['h_bar_ev'] * flexSpectrum.phys_const['c'])
        
        # Frequency square:
        w2 = _w2_space_(shape, pixelsize)
        
        return 1 - alpha * w2 * r_eff / (2*k)
       
def _w_space_(shape, dim, pixelsize):
    """
    Generate spatial frequencies along dimension dim.
    """                   
    # Image dimentions:
    sz = numpy.array(shape) * pixelsize
        
    # Frequency:
    xx = numpy.arange(0, shape[dim]) - shape[dim]//2
    return 2 * numpy.pi * xx / sz[dim]

def _w2_space_(shape, pixelsize):
    """
    Generates the lambda*freq**2*R image that can be used to calculate phase propagator at distance R, photon wavelength lambda.
    """
    # Frequency squared:
    u = _w_space_(shape, 0, pixelsize)
    v = _w_space_(shape, 1, pixelsize)
    return numpy.fft.fftshift((u**2)[:, None] + (v**2)[None, :])
        
def apply_ctf(image, ctf):
    """
    Apply CTF to the image using convolution.
    """
    if image.ndim > 2:
        
        x = numpy.fft.fft2(image, axes = (0, 2)) * ctf
        x = numpy.abs(numpy.fft.ifft2( x , axes = (0, 2)))
        x = numpy.array(x, dtype = 'float32')
        return x
    
    else:        
        x = numpy.fft.fft(image) * ctf
        x = numpy.abs(numpy.fft.ifft2( x ))
        x = numpy.array(x, dtype = 'float32')
        return x

def deapply_ctf(image, ctf, epsilon = 0.1):
    """
    Inverse convolution with Tikhonov regularization.
    """
    if image.ndim > 2:
        
        x = numpy.fft.fft2(image, axes = (0, 2)) * numpy.conj(ctf) / (abs(ctf) ** 2 + epsilon)
        x = numpy.abs(numpy.fft.ifft2( x , axes = (0, 2)))
        x = numpy.array(x, dtype = 'float32')
        return x
    
    else:        
        x = numpy.fft.fft(image) * numpy.conj(ctf) / (abs(ctf) ** 2 + epsilon)
        x = numpy.abs(numpy.fft.ifft2( x ))
        x = numpy.array(x, dtype = 'float32')
        return x
        
def apply_noise(image, mode = 'poisson', parameter = 1):
    """
    Add noise to the data.
    
    Args:
        image (numpy.array): image to apply noise to
        mode (str): poisson or normal
        parameter (float): norm factor for poisson or a standard deviation    
    """
    
    if mode == 'poisson':
        return numpy.random.poisson(image * parameter)
        
    elif mode == 'normal':
        return numpy.random.normal(image, parameter)
        
    else: 
        raise ValueError('Me not recognize the mode! Use normal or poisson!')

def effective_spectrum(kv = 90, filtr = {'material':'Cu', 'density':8, 'thickness':0.1}, detector = {'material':'Si', 'density':5, 'thickness':1}):
    """
    Generate an effective specturm of a CT scanner.
    """            
    energy = numpy.linspace(10, 90, 9)
    
    # Tube:
    spectrum = flexSpectrum.bremsstrahlung(energy, kv) 
    
    # Filter:
    if filtr:
        spectrum *= flexSpectrum.total_transmission(energy, filtr['material'], rho = filtr['density'], thickness = filtr['thickness'])
    
    # Detector:
    if detector:    
        spectrum *= flexSpectrum.scintillator_efficiency(energy, detector['material'], rho = detector['density'], thickness = detector['thickness'])
    
    # Normalize:
    spectrum /= (energy*spectrum).sum()
    
    return energy, spectrum
    
def spectralize(proj, kv = 90, n_phot = 1e8, specimen = {'material':'Al', 'density': 2.7}, filtr = {'material':'Cu', 'density':8, 'thickness':0.1}, detector = {'material':'Si', 'density':5, 'thickness':1}):
    """
    Simulate spectral data.
    """
    
    # Generate spectrum:
    energy, spectrum = effective_spectrum(kv, filtr, detector)
    
    # Get the material refraction index:
    mu = flexSpectrum.linear_attenuation(energy, specimen['material'], specimen['density'])
     
    # Display:
    flexUtil.plot(energy, spectrum, title = 'Spectrum') 
    flexUtil.plot(energy, mu, title = 'Linear attenuation') 
        
    # Simulate intensity images:
    counts = numpy.zeros_like(proj)
        
    for ii in range(len(energy)):
        
        # Monochromatic component:
        monochrome = spectrum[ii] * numpy.exp(-proj * mu[ii])
        monochrome = apply_noise(monochrome, 'poisson', n_phot) / n_phot    
        
        # Detector response is assumed to be proportional to E
        counts += energy[ii] * monochrome
    
    return counts

def forward_spectral(vol, proj, geometry, materials, energy, spectrum, n_phot = 1e8):
    """
    Simulate spectral data using labeled volume.
    """
    
    max_label = int(vol.max())
    
    if max_label != len(materials): raise ValueError('Number of materials is not the same as the number of labels in the volume!')

    # Normalize spectrum:
    spectrum /= (spectrum * energy).sum()
    
    # Simulate intensity images:
    lab_proj = []
    for jj in range(max_label):
        
        # Forward project:    
        proj_j = numpy.zeros_like(proj)
        vol_j = numpy.float32(vol == (jj+1))
        flexProject.forwardproject(proj_j, vol_j, geometry)
        
        lab_proj.append(proj_j)
        
    for ii in range(len(energy)):
        
        # Monochromatic components:
        monochrome = numpy.ones_like(proj)
        
        for jj in range(max_label):
            
            mu = flexSpectrum.linear_attenuation(energy, materials[jj]['material'], materials[jj]['density'])
    
            monochrome *= numpy.exp(-lab_proj[jj] * mu[ii])
            
        monochrome *= spectrum[ii]
        monochrome = apply_noise(monochrome, 'poisson', n_phot) / n_phot    

        # Detector response is assumed to be proportional to E
        proj += energy[ii] * monochrome