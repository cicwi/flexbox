#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in Nov 2017

@author: kostenko

This module uses NIST data (embedded in xraylib module) to simulate x-ray spectra of compounds.

"""

import numpy
import xraylib

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

def linear_attenuation(energy, compound, rho, thickness):
    '''
    Total X-ray absorption for a given compound in cm2g. Energy is given in KeV
    '''
    # xraylib might complain about types:
    energy = numpy.double(energy)        
    
    return thickness * rho * mass_attenuation(energy, compound)
    
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
    
def scintillator_efficiency(energy, compound = 'BaFBr', rho = 5, thickness = 100):
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
    return numpy.exp(-linear_attenuation(energy, compound, rho, thickness))

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
    