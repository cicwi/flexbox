#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in Nov 2017

@author: kostenko

This module uses NIST data (embedded in xraylib module) to simulate x-ray spectra of compounds.

"""

import numpy

# Some useful physical constants:
phys_const = {'c': 299792458, 'h': 6.62606896e-34, 'h_ev': 4.13566733e-15, 'h_bar': 1.054571628e-34, 'h_bar_ev': 6.58211899e-16,
              'e': 1.602176565e-19, 'Na': 6.02214179e23, 're': 2.817940289458e-15, 'me': 9.10938215e-31, 'me_ev': 0.510998910e6}
const_unit = {'c': 'm/c', 'h': 'J*S', 'h_ev': 'e*Vs', 'h_bar': 'J*s', 'h_bar_ev': 'eV*s',
              'e': 'colomb', 'Na': '1/mol', 're': 'm', 'me': 'kg', 'me_ev': 'ev/c**2'}


def material_refraction(energy, compound, rho):
    """    
    Calculate complex refrative index of the material taking
    into account it's density. 

    Args:
        compound (str): compound chemical formula
        rho (float): density in g / cm3
        energy (numpy.array): energy in KeV   

    Returns:
        float: refraction index in [1/mm]
    """
    
    import xraylib

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
    wavelength = 2 * numpy.pi * \
        (phys_const['h_bar_ev'] * phys_const['c']) / energy * 10

    # TODO: check this against phantoms.m:
    phi = rho_e * phys_const['re'] * wavelength

    # Refraction index (per mm)
    return rho * (mu / 2 - 1j * phi) / 10


def mass_attenuation(energy, compound):
    '''
    Total X-ray absorption for a given compound in cm2g. Energy is given in KeV
    '''
    import xraylib
    
    # xraylib might complain about types:
    energy = numpy.double(energy)

    if numpy.size(energy) == 1:
        return xraylib.CS_Total_CP(compound, energy)
    else:
        return numpy.array([xraylib.CS_Total_CP(compound, e) for e in energy])


def linear_attenuation(energy, compound, rho):
    '''
    Total X-ray absorption for a given compound in 1/mm. Energy is given in KeV
    '''
    # xraylib might complain about types:
    energy = numpy.double(energy)

    # unit: [1/mm]
    return rho * mass_attenuation(energy, compound) / 10


def compton(energy, compound):
    '''
    Compton scaterring crossection for a given compound in cm2g. Energy is given in KeV
    '''
    
    # xraylib might complain about types:
    energy = numpy.double(energy)
    import xraylib
    
    if numpy.size(energy) == 1:
        return xraylib.CS_Compt_CP(compound, energy)
    else:
        return numpy.array([xraylib.CS_Compt_CP(compound, e) for e in energy])


def rayleigh(energy, compound):
    '''
    Compton scaterring crossection for a given compound in cm2g. Energy is given in KeV
    '''
    import xraylib
    
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
    import xraylib
    
    # xraylib might complain about types:
    energy = numpy.double(energy)

    if numpy.size(energy) == 1:
        return xraylib.CS_Photo_CP(compound, energy)
    else:
        return numpy.array([xraylib.CS_Photo_CP(compound, e) for e in energy])


def scintillator_efficiency(energy, compound='BaFBr', rho=5, thickness=1):
    '''
    Generate QDE of a detector (scintillator). Units: KeV, g/cm3, mm.
    '''
    # Attenuation by the photoelectric effect:
    spectrum = 1 - numpy.exp(- thickness * rho *
                             photoelectric(energy, compound) / 10)

    # Normalize:
    return spectrum / spectrum.max()


def total_transmission(energy, compound, rho, thickness):
    '''
    Compute fraction of x-rays transmitted through the filter. 
    Units: KeV, g/cm3, mm.
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
    return numpy.exp(-(energy - energy_mean)**2 / (2 * energy_sigma**2))


def nist_names():
    '''
    Get a list of registered compound names understood by nist
    '''
    import xraylib
    
    return xraylib.GetCompoundDataNISTList()


def find_nist_name(compound_name):
    '''
    Get physical properties of one of the compounds registered in nist database
    '''
    import xraylib
    
    return xraylib.GetCompoundDataNISTByName(compound_name)


def parse_compound(compund):
    '''
    Parse chemical formula
    '''
    import xraylib
    
    return xraylib.CompoundParser(compund)