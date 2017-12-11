#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 2017
@author: kostenko

This module some wrappers around ASTRA to make lives of people slightly less horrible.
"""

''' * Imports * '''

import numpy
import misc
import astra
import astra.experimental as asex 
import sys
import matplotlib.pyplot as plt
import random

import flexUtil
import flexData
import flexModel

''' * Methods * '''

def _backproject_block_(projections, volume, proj_geom, vol_geom, algorithm = 'BP3D_CUDA', operation = '+'):
    """
    Use this internal function to compute backprojection of a single block of data.
    """           
    try:
        
        if (operation == '+'):
            volume_ = volume
            
        elif (operation == '*') | (operation == '/'):
            volume_ = numpy.zeros_like(volume)
            
        else: ValueError('Unknown operation type!')
                    
        sin_id = astra.data3d.link('-sino', proj_geom, projections)        
        vol_id = astra.data3d.link('-vol', vol_geom, volume_)    
        
        projector_id = astra.create_projector('cuda3d', proj_geom, vol_geom)
    
        if algorithm == 'BP3D_CUDA':
            asex.accumulate_BP(projector_id, vol_id, sin_id)
            
        elif algorithm == 'FDK_CUDA':
            asex.accumulate_FDK(projector_id, vol_id, sin_id)
            
        else:
            raise ValueError('Unknown ASTRA algorithm type.')
        
        if (operation == '*'):
             volume *= volume_
        elif (operation == '/'):
             volume_[volume_ < 1e-10] = numpy.inf
             volume /= volume_
             
    except:
        print("ASTRA error:", sys.exc_info())
        
    finally:
        astra.algorithm.delete(projector_id)
        astra.data3d.delete(sin_id)
        astra.data3d.delete(vol_id)                 
            
def _forwardproject_block_(projections, volume, proj_geom, vol_geom, operation = '+'):
    """
    Use this internal function to compute backprojection of a single block of data.
    """           
    try:
        
        if (operation == '+'):
            projections_ = projections
            
        elif (operation == '*') | (operation == '/'):
            projections_ = numpy.zeros_like(projections)
            
        else: ValueError('Unknown operation type!')    
                
        sin_id = astra.data3d.link('-sino', proj_geom, projections_)        
        vol_id = astra.data3d.link('-vol', vol_geom, volume)    
        
        projector_id = astra.create_projector('cuda3d', proj_geom, vol_geom)
        
        asex.accumulate_FP(projector_id, vol_id, sin_id)
        
        if (operation == '*'):
             projections *= projections_
        elif (operation == '/'):
            
             projections_[projections_ < 1e-10] = numpy.inf        
             projections /= projections_
             
    except:
        print("ASTRA error:", sys.exc_info())
        
    finally:
        astra.algorithm.delete(projector_id)
        astra.data3d.delete(sin_id)
        astra.data3d.delete(vol_id)            
            
def backproject(projections, volume, geometry, algorithm = 'BP3D_CUDA', operation = '+'):
    """
    Backproject useing standard ASTRA functionality
    """
    # If the data is not memmap:        
    if not isinstance(projections, numpy.memmap):    
        
        projections = numpy.ascontiguousarray(projections) 
        
        # Initialize ASTRA geometries:
        vol_geom = flexData.astra_vol_geom(geometry, volume.shape)
        proj_geom = flexData.astra_proj_geom(geometry, projections.shape[::2])    
        
        _backproject_block_(projections, volume, proj_geom, vol_geom, algorithm, operation)
        
    else:
        # Decide on the size of the block:
        n = projections.shape[1]    
        l = n // 10
        
        # Initialize ASTRA geometries:
        vol_geom = flexData.astra_vol_geom(geometry, volume.shape)
        
        # Loop over blocks:
        for ii in range(n // l):
            
            i0 = (ii * l)
            i1 = min((ii * l + l), n+1)
            
            # Extract a block:
            proj_geom = flexData.astra_proj_geom(geometry, projections.shape[::2], numpy.arange(i0, i1))    
            
            block = numpy.ascontiguousarray(projections[:, i0:i1,:])
            
            # Backproject:    
            _backproject_block_(block, volume, proj_geom, vol_geom, algorithm, operation)  

            
def forwardproject(projections, volume, geometry, operation = '+'):
    """
    Forwardproject
    """
    # If the data is not memmap:        
    if not isinstance(volume, numpy.memmap):   
        
        # Initialize ASTRA geometries:
        vol_geom = flexData.astra_vol_geom(geometry, volume.shape)
        proj_geom = flexData.astra_proj_geom(geometry, projections.shape[::2])
        
        _forwardproject_block_(projections, volume, proj_geom, vol_geom, operation)
        
    else:
        
        # Decide on the size of the block:
        n = volume.shape[0]    
        l = n // 10
        
        # Initialize ASTRA geometries:
        proj_geom = flexData.astra_proj_geom(geometry, projections.shape[::2])    
        
        # Loop over blocks:
        for ii in range(n // l):
            
            i0 = (ii * l)
            i1 = min((ii * l + l), n+1)
            
            # Extract a block:
            vol_geom = flexData.astra_vol_geom(geometry, volume.shape, i0, i1)
            
            block = numpy.ascontiguousarray(projections[i0:i1, :, :])

            # Forwardproject:
            _forwardproject_block_(block, volume, proj_geom, vol_geom, operation)  
            
def init_volume(projections):
    """
    Initialize a standard volume array.
    """          
    shape = projections.shape
    return numpy.zeros([shape[0], shape[2], shape[2]], dtype = 'float32')
    
def sample_FDK(projections, geometry, sample):
    """
    Quick reconstruction of a subsampled version of FDK
    """
    
    # Standard volume:
    #backproject(projections, volume, geometry, 'FDK_CUDA')
    # Adapt the geometry to the subsampling level:
    projections_ = numpy.ascontiguousarray(projections[::sample[0], :, ::sample[1]]) 
    volume = init_volume(projections_)
    
    # Initialize ASTRA geometries:
    vol_geom = flexData.astra_vol_geom(geometry, volume.shape, sample = sample)
    proj_geom = flexData.astra_proj_geom(geometry, projections_.shape[::2], sample = sample)    
    
    _backproject_block_(projections_, volume, proj_geom, vol_geom, 'FDK_CUDA')
    
    return volume
    
def FDK(projections, volume, geometry):
    """
    FDK
    """
    # Make sure array is contiguous (if not memmap):
    misc.progress_bar(0)    
    
    # Yeeey!
    backproject(projections, volume, geometry, 'FDK_CUDA')
    
    misc.progress_bar(1) 
    
def _block_index_(ii, block_number, length, mode = None):
    """
    Create a slice for a projection block
    """   
    
    # Length of the block and the global index:
    block_length = int(numpy.round(length / block_number))
    index = numpy.arange(length)

    # Different indexing modes:    
    if (mode == 'sequential')|(mode is None):
        # Index = 0, 1, 2, 4
        pass
        
    elif mode == 'random':   
        # Index = 2, 3, 0, 1 for instance...        
        random.shuffle(index)    
         
    elif mode == 'equidistant':   
        
        # Index = 0, 2, 1, 3   
        index = numpy.mod(numpy.arange(length) * block_length, length)
        
    else:
        raise ValueError('Indexer type not recognized! Use: sequential/random/equidistant')
        
    first = ii * block_length
    last = min((length + 1, (ii + 1) * block_length))
    
    return index[first:last]
    
def _L2_step_(projections, prj_weight, volume, geometry, options, operation = '+'):
    """
    Update volume: single SIRT step.
    """
    
    # CTF, mode of indexing:
    ctf = options.get('ctf')
    mode = options.get('mode')
    
    # How many blocks?    
    block_number = options.get('block_number')
    if block_number is None: block_number = 1
    
    # Force block number if array is numpy.memmap
    if isinstance(projections, numpy.memmap):
        block_number  = max((10, block_number))
        
    length = projections.shape[1]
    
    # Initialize ASTRA geometries:
    vol_geom = flexData.astra_vol_geom(geometry, volume.shape)      
    
    for ii in range(block_number):
        
        # Create index slice to address projections:
        index = _block_index_(ii, block_number, length, mode)
        if index is []: break

        # Extract a block:
        proj_geom = flexData.astra_proj_geom(geometry, projections.shape[::2], index = index)    
        
        # Copy data to a block or simply pass a pointer to data itself if block is one.
        if (mode == 'sequential') & (block_number == 1):
            block = projections
            
        else:
            block = (projections[:, index, :])
            block = numpy.ascontiguousarray(block)
                
        if ctf is None:
            
            # Forwardproject:
            _forwardproject_block_(block, -volume, proj_geom, vol_geom, '+')   
            
        else:
            
            # Reserve memory for a forward projection (keep it separate because of CTF application):
            synth = numpy.ascontiguousarray(numpy.zeros_like(block))
  
            # Forwardproject:
            _forwardproject_block_(synth, volume, proj_geom, vol_geom, '+')   
            
            # CTF can be applied to each projection separately:
            synth = flexModel.apply_ctf(synth, ctf)

            # Compute residual:        
            block = (block - synth)
        
        # Take into account Poisson:
        if options.get('poisson_weight'):
            # Some formula representing the effect of photon starvation...
            block *= numpy.sqrt(numpy.exp(-projections[:, index, :]))    
            
        block *= prj_weight * block_number
        
        # L2 norm (use the last block to update):
        if options.get('l2_update'):
            l2 = (numpy.sqrt((block ** 2).mean()))
            
        else:
            l2 = [] 
          
        # Project
        _backproject_block_(block, volume, proj_geom, vol_geom, 'BP3D_CUDA', operation)    
    
    # Apply bounds
    if options.get('bounds') is not None:
        numpy.clip(volume, a_min = options['bounds'][0], a_max = options['bounds'][1], out = volume) 

    return l2   

def _em_step_(projections, prj_weight, volume, geometry, options):
    """
    Update volume: single SIRT step.
    """
    
    # CTF, mode of indexing:
    ctf = options.get('ctf')
    mode = options.get('mode')
    
    # How many blocks?    
    block_number = options.get('block_number')
    if block_number is None: block_number = 1
    
    # Force block number if array is numpy.memmap
    if isinstance(projections, numpy.memmap):
        block_number  = max((10, block_number))
        
    length = projections.shape[1]
    
    # Initialize ASTRA geometries:
    vol_geom = flexData.astra_vol_geom(geometry, volume.shape)      
    
    for ii in range(block_number):
        
        # Create index slice to address projections:
        index = _block_index_(ii, block_number, length, mode)
        if index is []: break

        # Extract a block:
        proj_geom = flexData.astra_proj_geom(geometry, projections.shape[::2], index = index)    
        
        # Copy data to a block or simply pass a pointer to data itself if block is one.
        if (mode == 'sequential') & (block_number == 1):
            block = projections
        else:
            block = (projections[:, index, :])
        
        # Reserve memory for a forward projection (keep it separate):
        synth = numpy.ascontiguousarray(numpy.zeros_like(block))
        
        # Forwardproject:
        _forwardproject_block_(synth, volume, proj_geom, vol_geom, '+')   
  
        # CTF can be applied to each projection separately:
        if ctf is not None:
            synth = flexModel.apply_ctf(synth, ctf)

        # Compute residual:        
        synth[synth < 1e-12] = numpy.inf  
        synth = (block / synth)
            
        # L2 norm (use the last block to update):
        if options.get('l2_update'):
            l2 = (numpy.sqrt((synth ** 2).mean()))
            
        else:
            l2 = [] 
          
        # Project
        _backproject_block_(synth * prj_weight * block_number, volume, proj_geom, vol_geom, 'BP3D_CUDA', '*')    
    
    # Apply bounds
    if options.get('bounds') is not None:
        numpy.clip(volume, a_min = options['bounds'][0], a_max = options['bounds'][1], out = volume) 

    return l2       
    
def SIRT(projections, volume, geometry, iterations, options = {'poisson_weight': False, 'l2_update': True, 'preview':False, 'bounds':None, 'block_number':1, 'index':'sequential', 'ctf': None}):
    """
    SIRT
    CTF is only applied in the blocky version of SIRT!
    """ 
    # Make sure array is contiguous (if not memmap):
    #if not isinstance(projections, numpy.memmap):
    #    projections = numpy.ascontiguousarray(projections)        
    
    # We will use quick and dirty scaling coefficient instead of proper calculation of weights
    m = (geometry['src2obj'] + geometry['det2obj']) / geometry['src2obj']
    prj_weight = 1 / (projections.shape[1] * (geometry['det_pixel'] / m) ** 4 * max(volume.shape))
                    
    # Initialize L2:
    l2 = []

    print('Doing SIRT`y things...')
    
    misc.progress_bar(0)
        
    for ii in range(iterations):
    
        # Update volume:
        l2_  = _L2_step_(projections, prj_weight, volume, geometry, options)
        l2.append(l2_)
                    
        # Preview
        if options.get('preview'):
            flexUtil.display_slice(volume, dim = 0)
            
        misc.progress_bar((ii+1) / iterations)
        
    if options.get('l2_update'):   

         plt.figure(15)
         plt.plot(l2)
         plt.title('Residual L2')    
    
def EM(projections, volume, geometry, iterations, options = {'preview':False, 'bounds':None, 'block_number':1, 'index':'sequential', 'l2_update': True}):
    """
    Expectation Maximization
    """ 
    # Make sure array is contiguous (if not memmap):
    #if not isinstance(projections, numpy.memmap):
     #   projections = numpy.ascontiguousarray(projections)    
        
    # Make sure that the volume is positive:
    if volume.max() <= 0: 
        volume *= 0
        volume += 1
    elif volume.min() < 0: volume[volume < 0] = 0

    projections[projections < 0] = 0

    # Initialize L2:
    l2 = []
            
    print('Em Emm Emmmm...')
    
    misc.progress_bar(0)
        
    for ii in range(iterations):

        # Temp projection data
        #forwardproject(projections, volume, geometry, operation = '/')
                
        # Temp reconstruction volume        
        #backproject(projections, volume, geometry, 'BP3D_CUDA', operation = '*')    
        
        # Update volume:
        l2_  = _em_step_(projections, 1, volume, geometry, options)
        l2.append(l2_)
                    
        # Preview
        if options.get('preview'):
            flexUtil.display_slice(volume, dim = 0)
                        
        misc.progress_bar((ii+1) / iterations)
        
    if options.get('l2_update'):

         plt.figure(15)
         plt.plot(l2)
         plt.title('Residual L2')