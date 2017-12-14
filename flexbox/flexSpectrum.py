#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in Nov 2017

@author: kostenko

This module uses NIST data (embedded in xraylib module) to simulate x-ray spectra of compounds.

"""

import numpy
import xraylib
import matplotlib.pyplot as plt

from . import misc
from . import flexUtil
from . import flexProject


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


def calibrate_spectrum(projections, volume, geometry, compound='Al', density=2.7, threshold=None, iterations=1000, n_bin=10):
    '''
    Use the projection stack of a homogeneous object to estimate system's 
    effective spectrum.
    Can be used by process.equivalent_thickness to produce an equivalent 
    thickness projection stack.
    Please, use conventional geometry. 
    '''

    # Find the shape of the object:
    segmentation = numpy.zeros_like(volume)

    if threshold:
        segmentation = numpy.array(volume > threshold, 'float32')
    else:
        max_ = numpy.percentile(volume, 99)
        segmentation = numpy.array(volume > (max_ / 2), 'float32')

    # Crop:
    height = segmentation.shape[0]
    w = 5
    segmentation = segmentation[height // 2 - w:height // 2 + w, :, :]
    projections_ = projections[height // 2 - w:height // 2 + w, :, :]

    flexUtil.display_slice(segmentation, title='segmentation')

    # Reprojected length:
    length = numpy.zeros_like(projections)

    length = length[height // 2 - w:height // 2 + w, :, :]

    # Forward project the shape:
    print('Calculating the attenuation length.')
    length = numpy.ascontiguousarray(length)

    flexProject.forwardproject(length, segmentation, geometry)

    #import flexModel
    #ctf = flexModel.get_ctf(length.shape[::2], 'gaussian', [1, 1])
    #length = flexModel.apply_ctf(length, ctf)

    # TODO: Some cropping might be needed to avoid artefacts at the edges

    flexUtil.display_slice(length, title='length sinogram')
    flexUtil.display_slice(projections_, title='apparent sinogram')

    length = length.ravel()
    intensity = numpy.exp(-projections_.ravel())

    lmax = length.max()
    lmin = length.min()

    print('Maximum reprojected length:', lmax)
    print('Minimum reprojected length:', lmin)

    print('Number of intensity-length pairs:', length.size)

    print('Computing the intensity-length transfer function.')

    # Bin number for lengthes:
    bin_n = 256
    bins = numpy.linspace(lmin, lmax, bin_n)

    # REbin:
    idx = numpy.digitize(length, bins)

    # Rebin length and intensity:
    length_0 = bins - (bins[1] - bins[0]) / 2
    intensity_0 = [numpy.median(intensity[idx == k]) for k in range(bin_n)]

    intensity_0 = numpy.array(intensity_0)
    length_0 = numpy.array(length_0)

    # Get rid of nans and more than 1 values:
    length_0 = length_0[intensity_0 < 0.99]
    intensity_0 = intensity_0[intensity_0 < 0.99]

    # Enforce zero-one values:
    length_0 = numpy.insert(length_0, 0, 0)
    intensity_0 = numpy.insert(intensity_0, 0, 1)

    # Get rid of tales:
    length_0 = length_0[5:-5]
    intensity_0 = intensity_0[5:-5]

    print('Intensity-length curve rebinned.')

    print('Computing the spectrum by Expectation Maximization.')

    energy = numpy.linspace(5, 100, n_bin)

    nb_iter = iterations
    mu = linear_attenuation(energy, compound, density)
    exp_matrix = numpy.exp(-numpy.outer(length_0, mu))

    # Initial guess of the spectrum:
    spec = bremsstrahlung(energy, 90)
    # Filter:
    spec *= total_transmission(energy, 'Cu', 8, 1)

    # Detector:
    spec *= scintillator_efficiency(energy, 'Si', rho=5, thickness=0.5)

    # Normalize:
    spec /= (energy * spec).sum()

    #spec = numpy.ones_like(energy)
    #spec[0] = 0
    #spec[-1] = 0

    norm_sum = exp_matrix.sum(0)

    spec0 = spec.copy()

    for iter in range(nb_iter):
        spec = spec * \
            exp_matrix.T.dot(intensity_0 / exp_matrix.dot(spec)) / norm_sum

        # Make sure that the total count of spec is 1
        spec = spec / spec.sum()

    print('Spectrum computed.')

    # synthetic intensity for a check:
    _intensity = exp_matrix.dot(spec)

    # Display:
    plt.figure()
    plt.scatter(length[::100], intensity[::100], color='k', alpha=.2, s=2)
    plt.plot(length_0, intensity_0, 'r:', lw=4, alpha=.8)
    plt.plot(length_0, _intensity, 'g-', lw=2, alpha=.6)
    plt.axis('tight')
    plt.title('Intensity v.s. absorption length.')
    plt.show()

    # Display:
    plt.figure()
    plt.plot(energy, spec, 'b')
    plt.plot(energy, spec0 * 100, 'r:')
    plt.axis('tight')
    plt.title('Calculated spectrum')
    plt.show()

    return energy, spec


def calibrate_spectrum_nobin(projections, volume, geometry, compound='Al', density=2.7, threshold=None, iterations=1000, n_bin=10):
    '''
    Use the projection stack of a homogeneous object to estimate system's 
    effective spectrum.
    Can be used by process.equivalent_thickness to produce an equivalent 
    thickness projection stack.
    Please, use conventional geometry. 
    '''

    # Find the shape of the object:
    segmentation = numpy.zeros_like(volume)

    if threshold:
        segmentation = numpy.array(volume > threshold, 'float32')
    else:
        max_ = numpy.percentile(volume, 99)
        segmentation = numpy.array(volume > (max_ / 2), 'float32')

    # Crop:
    height = segmentation.shape[0]
    w = 5
    segmentation = segmentation[height // 2 - w:height // 2 + w, :, :]
    projections_ = projections[height // 2 - w:height // 2 + w, :, :]

    flexUtil.display_slice(segmentation, title='segmentation')

    # Reprojected length:
    length = numpy.zeros_like(projections)

    length = length[height // 2 - w:height // 2 + w, :, :]

    # Forward project the shape:
    print('Calculating the attenuation length.')
    length = numpy.ascontiguousarray(length)

    flexProject.forwardproject(length, segmentation, geometry)

    #import flexModel
    #ctf = flexModel.get_ctf(length.shape[::2], 'gaussian', [1, 1])
    #length = flexModel.apply_ctf(length, ctf)

    # TODO: Some cropping might be needed to avoid artefacts at the edges

    flexUtil.display_slice(length, title='length sinogram')
    flexUtil.display_slice(projections_, title='apparent sinogram')

    length = length.ravel()
    intensity = numpy.exp(-projections_.ravel())

    lmax = length.max()
    lmin = length.min()

    print('Maximum reprojected length:', lmax)
    print('Minimum reprojected length:', lmin)

    print('Number of intensity-length pairs:', length.size)

    print('Computing the spectrum by Expectation Maximization.')

    energy = numpy.linspace(5, 100, n_bin)

    nb_iter = iterations
    mu = linear_attenuation(energy, compound, density)
    exp_matrix = numpy.exp(-numpy.outer(length, mu))

    spec = numpy.ones_like(energy)

    norm_sum = exp_matrix.sum(0)

    for iter in range(nb_iter):
        spec = spec * \
            exp_matrix.T.dot(intensity / exp_matrix.dot(spec)) / norm_sum

        # Make sure that the total count of spec is 1
        spec = spec / spec.sum()

    print('Spectrum computed.')

    # synthetic intensity for a check:
    _intensity = exp_matrix.dot(spec)

    # Display:
    plt.figure()
    plt.scatter(length[::100], intensity[::100], color='k', alpha=.2, s=2)
    plt.scatter(length[::200], _intensity[::200], color='b', alpha=.2, s=2)
    plt.axis('tight')
    plt.title('Intensity v.s. absorption length.')
    plt.show()

    # Display:
    plt.figure()
    plt.plot(energy, spec, 'b')
    plt.axis('tight')
    plt.title('Calculated spectrum')
    plt.show()

    return energy, spec


def equivalent_density(projections, geometry, energy, spectrum, compound, density):
    '''
    Transfrom intensity values to projected density for a single material data
    '''
    # Assuming that we have log data!

    print('Generating the transfer function.')

    # Attenuation of 1 mm:
    mu = linear_attenuation(energy, compound, density)

    # Make thickness range that is sufficient for interpolation:
    m = (geometry['src2obj'] + geometry['det2obj']) / geometry['src2obj']
    img_pix = geometry['det_pixel'] / m

    thickness_min = 0
    thickness_max = max(projections.shape) * img_pix

    print('Assuming thickness range:', [thickness_min, thickness_max])
    thickness = numpy.linspace(
        thickness_min, thickness_max, max(projections.shape))

    exp_matrix = numpy.exp(-numpy.outer(thickness, mu))
    synth_counts = exp_matrix.dot(spectrum)

    synth_counts = -numpy.log(synth_counts)

    plt.figure()
    plt.plot(thickness, synth_counts, 'r-', lw=4, alpha=.8)
    plt.axis('tight')
    plt.title('Attenuation v.s. thickness [mm].')
    plt.show()

    print('Callibration attenuation range:', [
          synth_counts[0], synth_counts[-1]])
    print('Data attenuation range:', [projections.min(), projections.max()])

    print('Applying transfer function.')

    for ii in range(projections.shape[1]):

        projections[:, ii, :] = numpy.array(numpy.interp(
            projections[:, ii, :], synth_counts, thickness * density), dtype='float32')
        misc.progress_bar((ii + 1) / projections.shape[1])

    return projections
