#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in Nov 2017

@author: kostenko

This module uses NIST data (embedded in xraylib module) to simulate x-ray spectra of compounds.

"""

import numpy
import xraylib

# Some useful physical constants:
phys_const = {'c': 299792458, 'h':6.62606896e-34, 'h_ev':4.13566733e-15, 'h_bar':1.054571628e-34,'h_bar_ev': 6.58211899e-16, 
                  'e':1.602176565e-19, 'Na':6.02214179e23, 're': 2.817940289458e-15,'me':9.10938215e-31, 'me_ev':0.510998910e6}
const_unit = {'c': 'm/c', 'h':'J*S', 'h_ev':'e*Vs', 'h_bar':'J*s', 'h_bar_ev':'eV*s' , 'e':'colomb', 'Na':'1/mol', 're':'m','me':'kg', 'me_ev':'ev/c**2'}

def material_refraction(compound, rho, energy):
    """    
    Calculate complex refrative index of the material taking
    into account it's density. 
    
    Args:
        compound (str): compound chemical formula
        rho (float): density in g / cm3
        energy (numpy.array): energy in KeV   
        
    Returns:
        float: refraction index in [1/cm]
    """
    
    cmp = xraylib.CompoundParser(compound)

    # Compute ration of Z and A:
    z = (numpy.array(cmp['Elements']))
    a = [xraylib.AtomicWeight(x) for x in cmp['Elements']]
    
    za = ((z / a) * numpy.array(cmp['massFractions'])).sum()
    
    # Electron density of the material:    
    Na = phys_const['Na']
    rho_e = rho * za * Na
    
    # Attenuation:
    mu = mass_attenuation(energy, compound)
    
    # Phase:
    wavelength = 2 * numpy.pi * (phys_const['h_bar_ev'] * phys_const['c']) / energy * 10   
                                
    # TODO: check this against phantoms.m:                            
    phi = rho_e * phys_const['re'] * wavelength
                    
    # Refraction index (per cm)
    return rho * (mu/2 - 1j * phi)
                       
def mass_attenuation(energy, compound):
    '''
    Total X-ray absorption for a given compound in cm2g. Energy is given in KeV
    '''
    
    # xraylib might complain about types:
    energy = numpy.double(energy)
    
    if numpy.size(energy) == 1:
        return xraylib.CS_Total_CP(compound, energy)   
    else:
        return numpy.array([xraylib.CS_Total_CP(compound, e) for e in energy])

def linear_attenuation(energy, compound, rho):
    '''
    Total X-ray absorption for a given compound in 1/cm. Energy is given in KeV
    '''
    # xraylib might complain about types:
    energy = numpy.double(energy)        
    
    return rho * mass_attenuation(energy, compound)
    
def compton(energy, compound):    
    '''
    Compton scaterring crossection for a given compound in cm2g. Energy is given in KeV
    '''
    
    # xraylib might complain about types:
    energy = numpy.double(energy)
    
    if numpy.size(energy) == 1:
        return xraylib.CS_Compt_CP(compound, energy)   
    else:
        return numpy.array([xraylib.CS_Compt_CP(compound, e) for e in energy])

def rayleigh(energy, compound):
    '''
    Compton scaterring crossection for a given compound in cm2g. Energy is given in KeV
    '''
    
    # xraylib might complain about types:
    energy = numpy.double(energy)
    
    if numpy.size(energy) == 1:
        return xraylib.CS_Rayl_CP(compound, energy)   
    else:
        return numpy.array([xraylib.CS_Rayl_CP(compound, e) for e in energy])
    
def photoelectric(energy, compound):    
    '''
    Photoelectric effect for a given compound in cm2g. Energy is given in KeV
    '''
    
    # xraylib might complain about types:
    energy = numpy.double(energy)
    
    if numpy.size(energy) == 1:
        return xraylib.CS_Photo_CP(compound, energy)   
    else:
        return numpy.array([xraylib.CS_Photo_CP(compound, e) for e in energy])
    
def scintillator_efficiency(energy, compound = 'BaFBr', rho = 5, thickness = 0.1):
    '''
    Generate QDE of a detector (scintillator). Units: KeV, g/cm3, cm.
    '''              
    # Attenuation by the photoelectric effect:
    spectrum = 1 - numpy.exp(- thickness * rho * photoelectric(energy, compound))
        
    # Normalize:
    return spectrum / spectrum.max()

def total_transmission(energy, compound, rho, thickness):
    '''
    Compute fraction of x-rays transmitted through the filter. 
    Units: KeV, g/cm3, cm.
    '''        
    # Attenuation by the photoelectric effect:
    return numpy.exp(-linear_attenuation(energy, compound, rho) * thickness)

def bremsstrahlung(energy, energy_max):
    '''
    Simple bremstrahlung model (Kramer formula). Emax
    '''
    spectrum = energy * (energy_max - energy)
    spectrum[spectrum < 0] = 0
        
    # Normalize:
    return spectrum / spectrum.max()

def gaussian_spectrum(energy, energy_mean, energy_sigma):
    '''
    Generates gaussian-like spectrum with given mean and STD.
    '''
    return numpy.exp(-(energy - energy_mean)**2 / (2*energy_sigma**2))
            
def nist_names():
    '''
    Get a list of registered compound names understood by nist
    '''
    return xraylib.GetCompoundDataNISTList()
        
def find_nist_name(compound_name):    
    '''
    Get physical properties of one of the compounds registered in nist database
    '''
    return xraylib.GetCompoundDataNISTByName(compound_name)

def parse_compound(compund):
    '''
    Parse chemical formula
    '''
    return xraylib.CompoundParser(compund)

def calibrate_spectrum(projections, volume, geometry, compound = 'Al', density = 2.7, threshold = None, iterations = 100000):
    '''
    Use the projection stack of a homogeneous object to estimate system's 
    effective spectrum.
    Can be used by process.equivalent_thickness to produce an equivalent 
    thickness projection stack.
    Please, use conventional geometry. 
    ''' 
    
    sz = projections.shape
    
    crop_proj = projections.copy()
    crop_vol = volume.copy()
    
    # Apply crop:
    window = 1   
    crop_proj = crop_proj[(sz[0]//2-window):(sz[0]//2+window), :, :]  
                                                
    sz = crop_vol.shape
    crop_vol = crop_vol[(sz[0]//2-window):(sz[0]//2+window), :, :]        
                                   
    # Find the shape of the object:                                                    
    #crop_vol.process.threshold(threshold = force_threshold)    
    # This way might not work because of mishandling of parents...                      
    if threshold:
        crop_vol = numpy.array(crop_vol > threshold, 'float32')
    else:
        crop_vol = numpy.array(crop_vol > (crop_vol.max()/2), 'float32')
      
    synth_proj = crop_proj.copy()
    synth_proj *= 0
    
    # Forward project the shape:                  
    print('Calculating the attenuation length.')  
    flexProject.forwardproject(synth_proj, crop_vol, geometry)
        
    # Projected length and intensity (only central slices):
    length = synth_proj[window//2:-window//2,:,:]
    intensity = crop_proj[window//2:-window//2,:,:]

    length = length.ravel()
    intensity = intensity.ravel()
    
    print('Maximum reprojected length:', length.max())
    print('Minimum reprojected length:', length.min())
    
    print('Number of intensity-length pairs:', length.size)
    
    print('Computing the intensity-length transfer function.')
    
    intensity = numpy.exp(-intensity)
    
    # Bin length (with half a pixel bin size):
    #max_len = length.max()    
    #bin_n = numpy.int(max_len * 2)
    bin_n = 1000
    
    bins = numpy.linspace(0.2, length.max() * 0.9, bin_n)
    idx  = numpy.digitize(length, bins)

    # Rebin length and intensity:        
    length_0 = bins - (bins[1]-bins[0]) / 2
    intensity_0 = [numpy.median(intensity[idx==k]) for k in range(bin_n)]

    intensity_0 = numpy.array(intensity_0)
    length_0 = numpy.array(length_0)
    
    # Get rid of nans and more than 1 values:
    length_0 = length_0[intensity_0 < 1]
    intensity_0 = intensity_0[intensity_0 < 1]
    
    # Enforce zero-one values:
    length_0 = numpy.insert(length_0, 0, 0)
    intensity_0 = numpy.insert(intensity_0, 0, 1)
    
    print('Intensity-length curve rebinned.')
    
    # Display:
    plt.figure()
    plt.scatter(length[::100], intensity[::100], color='k', alpha=.2, s=2)
    plt.plot(length_0, intensity_0, 'r--', lw=4, alpha=.8)
    plt.axis('tight')
    plt.title('Intensity v.s. absorption length.')
    plt.show() 
    
    print('Computing the spectrum by Expectation Maximization.')
    
    print('Number of length bins:', intensity_0.size)
    print('Number of energy bins:', energy.size)
    
    nb_iter = iterations
    mu = spectra.linear_attenuation(energy, compound, density) * 0.1
    exp_matrix = numpy.exp(-numpy.outer(length_0, mu))

    spec = numpy.ones_like(energy)
    
    norm_sum = exp_matrix.sum(0)
    
    for iter in range(nb_iter): 
        spec = spec * exp_matrix.T.dot(intensity_0 / exp_matrix.dot(spec)) / norm_sum

        # Make sure that the total count of spec is 1
        spec = spec / spec.sum()

    print('Spectrum computed.')
         
    plt.figure()
    plt.plot(energy, spec) 
    plt.title('Calculated spectrum.')
        
    i_synthetic = exp_matrix.dot(spec)
    
    # Synthetic spread of Al:
    plt.figure()
    plt.plot(length_0, intensity_0) 
    plt.plot(length_0, i_synthetic) 
    plt.legend(['Measured intensity','Synthetic intensity'])
    
    return energy, spec, length_0, intensity_0
