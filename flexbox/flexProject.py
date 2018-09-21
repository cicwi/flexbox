#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 2017
@author: kostenko

This module some wrappers around ASTRA to make lives of people slightly less horrible.
"""

''' * Imports * '''

import numpy
import astra
import sys
import matplotlib.pyplot as plt
import random
import scipy 

from . import flexUtil
from . import flexData
from . import flexModel

''' * Methods * '''

def misfit(res, scl, deg):
    
    c = -numpy.size(res) * (scipy.special.gammaln((deg + 1) / 2) - 
            scipy.special.gammaln(deg / 2) - .5 * numpy.log(numpy.pi*scl*deg))
    
    return c + .5 * (deg + 1) * sum(numpy.log(1 + numpy.conj(res) * res / (scl * deg)))
    
def st(res, scl, deg):   
    
    grad = numpy.float32(scl * (deg + 1) * res / (scl * deg + numpy.conj(res) * res))
    
    return grad
    
def studentst(res, deg = 1, scl = None):
    
    # nD to 1D:
    shape = res.shape
    res = res.ravel()
    
    # Optimize scale:
    if scl is None:    
        fun = lambda x: misfit(res[::70], x, deg)
        scl = scipy.optimize.fmin(fun, x0 = [1,], disp = 0)[0]
        #scl = numpy.percentile(numpy.abs(res), 90)
        #print(scl)
        #print('Scale in Student`s-T is:', scl)
        
    # Evaluate:    
    grad = numpy.reshape(st(res, scl, deg), shape)
    
    return grad

def _sanity_check_data_(data):
    """
    Check if this data array is OK for ASTRA.
    """
    
    if min(data.shape) == 0 | (data.dtype != 'float32') | (~numpy.isfinite(data)).any():
        
        raise TypeError('Data is corrupted! Data:', data)

def _backproject_block_(projections, volume, proj_geom, vol_geom, algorithm = 'BP3D_CUDA', operation = '+'):
    """
    Use this internal function to compute backprojection of a single block of data.
    """           
    
    # Unfortunately need to hide the experimental ASTRA
    import astra.experimental as asex 
    import traceback
    
    _sanity_check_data_(projections)
    _sanity_check_data_(volume)
    
    try:
        
        if (operation == '+'):
            volume_ = volume
            
        elif (operation == '*') | (operation == '/'):
            volume_ = numpy.zeros_like(volume)
            
        else: ValueError('Unknown operation type!')
        
        if (operation == '-'):
            projections *= -1
                    
        sin_id = astra.data3d.link('-sino', proj_geom, projections)        
        vol_id = astra.data3d.link('-vol', vol_geom, volume_)    
        
        projector_id = astra.create_projector('cuda3d', proj_geom, vol_geom)
        
        if algorithm == 'BP3D_CUDA':
            asex.accumulate_BP(projector_id, vol_id, sin_id)
            
        elif algorithm == 'FDK_CUDA':
            asex.accumulate_FDK(projector_id, vol_id, sin_id)
            
        else:
            raise ValueError('Unknown ASTRA algorithm type.')
        
        if (operation == '-'):
            projections *= -1            
            
        elif (operation == '*'):
            
             volume *= volume_
            
             #flexUtil.display_slice(volume, dim = 0, title = 'worm')
            
             # This is really slow but needed in case of overlap for EM: 
             volume_ *= 0
             astra.data3d.delete(sin_id)
             sin_id = astra.data3d.link('-sino', proj_geom, projections * 0 + 1)
             
             asex.accumulate_BP(projector_id, vol_id, sin_id)
             
             #flexUtil.display_slice(volume_, dim = 0, title = 'norm')
             
             volume_[volume_ < 0.01] = 0.01
             
             volume /= volume_
             
             #flexUtil.display_slice(volume, dim = 0, title = 'rorm')
             
             
        elif (operation == '/'):
             volume_[volume_ < 1e-3] = numpy.inf
             volume /= volume_
             
    except:
        #print("ASTRA error:", sys.exc_info())
        info = sys.exc_info()
        traceback.print_exception(*info)        
        
    finally:
        astra.algorithm.delete(projector_id)
        astra.data3d.delete(sin_id)
        astra.data3d.delete(vol_id)                 
            
def _forwardproject_block_(projections, volume, proj_geom, vol_geom, operation = '+'):
    """
    Use this internal function to compute backprojection of a single block of data.
    """           
    # Unfortunately need to hide the experimental ASTRA
    import astra.experimental as asex 
    import traceback
    
    _sanity_check_data_(projections)
    _sanity_check_data_(volume)
    
    try:
        
        if (operation == '+'):
            projections_ = projections
            
        elif (operation == '-'):
            projections_ = -projections
            
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
             
        elif (operation == '-'):
            projections[:] = -projections_[:]
             
    except:
        #print("ASTRA error:", sys.exc_info())
        info = sys.exc_info()
        traceback.print_exception(*info)
        
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
        print('FDK continu')
        
        projections = numpy.ascontiguousarray(projections) 
        
        # Initialize ASTRA geometries:
        vol_geom = flexData.astra_vol_geom(geometry, volume.shape)
        proj_geom = flexData.astra_proj_geom(geometry, projections.shape)    
        
        _backproject_block_(projections, volume, proj_geom, vol_geom, algorithm, operation)
        
    else:
        print('FDK blocky')
        # Decide on the size of the block:
        n = projections.shape[1]    
        l = n // 20
        
        # Initialize ASTRA geometries:
        vol_geom = flexData.astra_vol_geom(geometry, volume.shape)
        
        # Loop over blocks:
        for ii in range(n // l):
            
            i0 = (ii * l)
            i1 = min((ii * l + l), n+1)
            
            # Extract a block:
            proj_geom = flexData.astra_proj_geom(geometry, projections.shape, numpy.arange(i0, i1))    
            
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
        proj_geom = flexData.astra_proj_geom(geometry, projections.shape)
        
        _forwardproject_block_(projections, volume, proj_geom, vol_geom, operation)
        
    else:
        
        # Decide on the size of the block:
        n = volume.shape[0]    
        l = n // 10
        
        # Initialize ASTRA geometries:
        proj_geom = flexData.astra_proj_geom(geometry, projections.shape)    
        
        # Loop over blocks:
        for ii in range(n // l):
            
            i0 = (ii * l)
            i1 = min((ii * l + l), n+1)
            
            # Extract a block:
            vol_geom = flexData.astra_vol_geom(geometry, volume.shape, i0, i1)
            
            block = numpy.ascontiguousarray(projections[i0:i1, :, :])

            # Forwardproject:
            _forwardproject_block_(block, volume, proj_geom, vol_geom, operation)  
                     
def init_volume(projections, geometry = None):
    """
    Initialize a standard volume array.
    """          
    
    if geometry:
        sample = geometry['proj_sample']

        offset = int(abs(geometry['vol_tra'][2]) / geometry['img_pixel'] / sample[2])

    else:
        offset = 0
        
        sample = [1, 1, 1]

    shape = projections[::sample[0], ::sample[1], ::sample[2]].shape
    return numpy.zeros([shape[0], shape[2]+offset, shape[2]+offset], dtype = 'float32')
    
def sample_FDK(projections, geometry, sample = [1,1,1]):
    """
    Quick reconstruction of a subsampled version of FDK
    """
    
    # Standard volume:
    
    # Adapt the geometry to the subsampling level:
    volume = init_volume(projections, geometry)
    
    # Change sampling:

    # Initialize ASTRA geometries:
    #vol_geom = flexData.astra_vol_geom(geometry, volume.shape, sample = sample)
    #proj_geom = flexData.astra_proj_geom(geometry, projections_.shape, sample = sample)    
    geometry_ = geometry.copy()

    # Apply subsampling to detector and volume:    
    geometry_['anisotrpy'] = [sample[0], sample[1], sample[2]]
    geometry_['sample'] = sample

    FDK(projections, volume, geometry_)
    
    #_backproject_block_(projections_, volume, proj_geom, vol_geom, 'FDK_CUDA')
    
    # Apply correct scaling:
    #volume /= geometry['img_pixel']**4    
    
    return volume
    
def FDK(projections, volume, geometry):
    """
    FDK.
    """
    # TODO: make JW fix normaliztion in rotated volumes.
    print('FDK reconstruction...')
    
    # Sampling:
    samp = geometry['proj_sample']
    
    # Make sure array is contiguous (if not memmap):
    flexUtil.progress_bar(0)    
    
    if sum(samp) > 3:
        backproject(projections[::samp[0],::samp[1], ::samp[2]], volume, geometry, 'FDK_CUDA')
    else:
        backproject(projections, volume, geometry, 'FDK_CUDA')
    
    volume /= (numpy.prod(samp) * geometry['img_pixel'])**4
    
    flexUtil.progress_bar(1) 
            
def _block_index_(ii, block_number, length, mode = 'sequential'):
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

def _L2_step_ctf_(projections, prj_weight, volume, geometry, options, operation = '+'):
    """
    A CTF version of the L2 update step.
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
        proj_geom = flexData.astra_proj_geom(geometry, projections.shape, index = index)    
        
        # Copy data to a block or simply pass a pointer to data itself if block is one.
        if (mode == 'sequential') & (block_number == 1):
            block = projections.copy()
            
        else:
            block = (projections[:, index, :]).copy()
            block = numpy.ascontiguousarray(block)
        
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
            #block *= numpy.sqrt(numpy.exp(-projections[:, index, :]))               
            block *= numpy.exp(-projections[:, index, :])
            
        block *= prj_weight * block_number
        
        # Apply ramp to reduce boundary effects:
        block = block = flexData.ramp(block, 2, 5, mode = 'linear')
        block = block = flexData.ramp(block, 0, 5, mode = 'linear')
                
        # L2 norm (use the last block to update):
        if options.get('l2_update'):
            l2 = (numpy.sqrt((block ** 2).mean()))
            
        else:
            l2 = 0 
          
        # Project
        _backproject_block_(block, volume, proj_geom, vol_geom, 'BP3D_CUDA', operation)    
    
    # Apply bounds
    if options.get('bounds') is not None:
        numpy.clip(volume, a_min = options['bounds'][0], a_max = options['bounds'][1], out = volume) 

    return l2   
    
def _L2_step_(projections, prj_weight, volume, geometry, options, operation = '+'):
    """
    Update volume: single SIRT step.
    """
    
    # Mode of indexing:
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
    
    l2 = 0
    
    for ii in range(block_number):
        
        # Create index slice to address projections:
        index = _block_index_(ii, block_number, length, mode)
        if index is []: break

        # Extract a block:
        proj_geom = flexData.astra_proj_geom(geometry, projections.shape, index = index)    
        
        if (mode == 'sequential') & (block_number == 1):
            block = projections.copy()
            #block = projections
            
        else:
            block = (projections[:, index, :]).copy()
            block = numpy.ascontiguousarray(block)
                
        # Forwardproject:
        _forwardproject_block_(block, volume, proj_geom, vol_geom, '-')   
                    
        # Take into account Poisson:
        if options.get('poisson_weight'):
            
            # Some formula representing the effect of photon starvation...
            block *= numpy.exp(-projections[:, index, :])
            
        block *= prj_weight * block_number
        
        # Apply ramp to reduce boundary effects:
        block = flexData.ramp(block, 0, 5, mode = 'linear')
        block = flexData.ramp(block, 2, 5, mode = 'linear')
                
        # L2 norm (use the last block to update):
        if options.get('l2_update'):
            l2 = (numpy.sqrt((block ** 2).mean()))
          
        # Project
        _backproject_block_(block, volume, proj_geom, vol_geom, 'BP3D_CUDA', operation)    
    
    # Apply bounds
    if options.get('bounds') is not None:
        numpy.clip(volume, a_min = options['bounds'][0], a_max = options['bounds'][1], out = volume) 

    return l2   
    
def _fista_step_(projections, prj_weight, vol, vol_old, vol_t, t, geometry, options):
    """
    Update volume: single SIRT step.
    """
    
    # Mode of indexing:
    mode = options.get('mode')
    
    # How many blocks?    
    block_number = options.get('block_number')
    if block_number is None: block_number = 1
    
    # Force block number if array is numpy.memmap
    if isinstance(projections, numpy.memmap):
        block_number  = max((10, block_number))
        
    length = projections.shape[1]
    
    # Initialize ASTRA geometries:
    vol_geom = flexData.astra_vol_geom(geometry, vol.shape)      
    
    vol_old[:] = vol.copy()  
    
    t_old = t 
    t = (1 + numpy.sqrt(1 + 4 * t**2))/2

    vol[:] = vol_t.copy()
    
    for ii in range(block_number):
        
        # Create index slice to address projections:
        index = _block_index_(ii, block_number, length, mode)
        if index is []: break

        # Extract a block:
        proj_geom = flexData.astra_proj_geom(geometry, projections.shape, index = index)    
        
        # Copy data to a block or simply pass a pointer to data itself if block is one.
        if (mode == 'sequential') & (block_number == 1):
            block = projections.copy()
            
        else:
            block = (projections[:, index, :]).copy()
            block = numpy.ascontiguousarray(block)
                
        # Forwardproject:
        _forwardproject_block_(block, vol_t, proj_geom, vol_geom, '-')   
                    
        # Take into account Poisson:
        if options.get('poisson_weight'):
            # Some formula representing the effect of photon starvation...
            block *= numpy.exp(-projections[:, index, :])
            
        block *= prj_weight * block_number
        
        # Apply ramp to reduce boundary effects:
        block = block = flexData.ramp(block, 2, 5, mode = 'linear')
        block = block = flexData.ramp(block, 0, 5, mode = 'linear')
                
        # L2 norm (use the last block to update):
        if options.get('l2_update'):
            l2 = (numpy.sqrt((block ** 2).mean()))
            
        else:
            l2 = 0 
          
        # Project
        _backproject_block_(block, vol, proj_geom, vol_geom, 'BP3D_CUDA', '+')   
        
        vol_t[:] = vol + ((t_old - 1) / t) * (vol - vol_old)
                
    # Apply bounds
    if options.get('bounds') is not None:
        numpy.clip(vol, a_min = options['bounds'][0], a_max = options['bounds'][1], out = vol) 

    return l2  
    
'''
# Forward:
    flex.project._forwardproject_block_(proj, vol_t, proj_geom, vol_geom, '-') 
    
    proj *= w
    
    vol_old = vol.copy()  
    t_old = t
    
    x = vol_t
    flex.project._backproject_block_(proj / L, x, proj_geom, vol_geom, 'BP3D_CUDA', '+')
        
    t = (1 + numpy.sqrt(1 + 4 * t**2))/2
    vol_t = x + ((t_old - 1) / t) * (vol - vol_old)
'''    

def _em_step_(projections, prj_weight, volume, geometry, options):
    """
    Update volume: single EM step.
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
        proj_geom = flexData.astra_proj_geom(geometry, projections.shape, index = index)    
        
        # Copy data to a block or simply pass a pointer to data itself if block is one.
        if (mode == 'sequential') & (block_number == 1):
            block = projections
            
        else:
            block = (projections[:, index, :]).copy()
        
        # Reserve memory for a forward projection (keep it separate):
        synth = numpy.ascontiguousarray(numpy.zeros_like(block))
        
        # Forwardproject:
        _forwardproject_block_(synth, volume, proj_geom, vol_geom, '+')   
  
        # CTF can be applied to each projection separately:
        if ctf is not None:
            synth = flexModel.apply_ctf(synth, ctf)

        # Compute residual:        
        synth[synth < 1e-10] = numpy.inf  
        synth = (block / synth)
                    
        # L2 norm (use the last block to update):
        if options.get('l2_update'):
            
            _synth = synth[synth > 0]
            l2 = _synth.std()
            
        else:
            l2 = [] 
          
        # Project
        _backproject_block_(synth * prj_weight * block_number, volume, proj_geom, vol_geom, 'BP3D_CUDA', '*')    
    
    # Apply bounds
    if options.get('bounds') is not None:
        numpy.clip(volume, a_min = options['bounds'][0], a_max = options['bounds'][1], out = volume) 

    return l2    
           
def SIRT(projections, volume, geometry, iterations, options = {'poisson_weight': False, 'l2_update': True, 'preview':False, 'bounds':None, 'block_number':10, 'mode':'sequential', 'ctf': None}):
    """
    SIRT
    CTF is only applied in the blocky version of SIRT!
    """     
    # Sampling:
    samp = geometry['proj_sample']
    anisotropy = geometry['vol_sample']

    #pix = max(samp) * geometry['img_pixel']
    pix = (geometry['img_pixel']**4 * anisotropy[0] * anisotropy[1] * anisotropy[2] * anisotropy[2])
    prj_weight = 1 / (projections[::samp[0], ::samp[1], ::samp[2]].shape[1] * pix * max(volume.shape)) 
                    
    # Initialize L2:
    l2 = []   

    print('Feeling SIRTy...')
    
    flexUtil.progress_bar(0)
        
    for ii in range(iterations):
    
        # Update volume:
        l2_  = _L2_step_(projections[::samp[0], ::samp[1], ::samp[2]], prj_weight, volume, geometry, options)
        l2.append(l2_)
                    
        # Preview
        if options.get('preview'):
            flexUtil.display_slice(volume, dim = 1)
            
        flexUtil.progress_bar((ii+1) / iterations)
        
    if options.get('l2_update'):   
         flexUtil.plot(l2, semilogy = True, title = 'Resudual L2')   
         
def FISTA(projections, volume, geometry, iterations, options = {'poisson_weight': False, 'l2_update': True, 'preview':False, 'bounds':None, 'block_number':10, 'mode':'sequential', 'ctf': None}):
    # Sampling:
    samp = geometry['proj_sample']
    anisotropy = geometry['vol_sample']
    
    pix = (geometry['img_pixel']**4 * anisotropy[0] * anisotropy[1] * anisotropy[2] * anisotropy[2])
    prj_weight = 1 / (projections[::samp[0], ::samp[1], ::samp[2]].shape[1] * pix * max(volume.shape)) 
                    
    # Initialize L2:
    l2 = []   
    t = 1
    
    volume_t = volume.copy()
    volume_old = volume.copy()

    print('FISTING in progress...')
    
    flexUtil.progress_bar(0)
        
    for ii in range(iterations):
    
        # Update volume:
        l2_  = _fista_step_(projections[::samp[0], ::samp[1], ::samp[2]], prj_weight, volume, volume_old, volume_t, t, geometry, options)
        l2.append(l2_)
        
        # Preview
        if options.get('preview'):
            flexUtil.display_slice(volume, dim = 0)
            
        flexUtil.progress_bar((ii+1) / iterations)
        
    if options.get('l2_update'):   
        flexUtil.plot(l2, semilogy = True, title = 'Resudual L2')   

def SIRT_tiled(projections, volume, geometries, iterations, options = {'poisson_weight': False, 'l2_update': True, 'preview':False, 'bounds':None, 'block_number':1, 'mode':'sequential', 'ctf': None}):
    """
    SIRT: tiled version.
    """ 
    
    # Make sure array is contiguous (if not memmap):
    # if not isinstance(projections, numpy.memmap):
    #    projections = numpy.ascontiguousarray(projections)        
    
    # Initialize L2:
    l2 = []
    
    # In tiled reconstructions position of the volume should be the same for each tile:
    geometries_ = []
    for geom in geometries:
        geom_ = geom.copy()
        
        # Compute average volume shift:
        geom_['vol_tra'] = numpy.mean([g['vol_tra'] for g in geometries], 0)
        geometries_.append(geom_)

    print('Doing SIRT`y things...')
    
    flexUtil.progress_bar(0)
        
    for ii in range(iterations):
        
        l2_ = 0
        for ii, proj in enumerate(projections):
            
            geom = geometries_[ii]

            #m = (geom['src2obj'] + geom['det2obj']) / geom['src2obj']
            # This weight is half of the normal weight to make sure convergence is ok:
            prj_weight = 1 / (proj.shape[1] * (geom['img_pixel']) ** 4 * max(volume.shape)) 
    
            # Update volume:
            l2_ += _L2_step_(proj, prj_weight, volume, geom, options)
            
        l2.append(l2_)
                    
        # Preview
        if options.get('preview'):
            flexUtil.display_slice(volume, dim = 0)
            
        flexUtil.progress_bar((ii+1) / iterations)
        
    if options.get('l2_update'):   
        flexUtil.plot(l2, semilogy = True, title = 'Resudual L2')      
         
def PWLS_M(projections, volume, geometries, n_iter = 10, block_number = 20, student = False, rings_t = 0, pwls = True, weight_power = 1): 
    '''
    Penalized Weighted Least Squares based on multiple inputs.
    '''
    #error log:
    L = []

    fac = volume.shape[2] * geometries[0]['img_pixel'] * numpy.sqrt(2)
    
    projsh= projections[0].shape[::2]

    print('PWLS-ing in progress...')
    flexUtil.progress_bar(0)
    
    # reconstruction volume:
    ring = numpy.zeros([projsh[0], projsh[1]], dtype = 'float32')
        
    # Iterations:
    for ii in range(n_iter):
    
        # Error:
        L_mean = 0
        
        #Blocks:
        for jj in range(block_number):        
            
            # Volume update:
            vol_tmp = numpy.zeros_like(volume)
            bwp_w = numpy.zeros_like(volume)
            
            for kk, projs in enumerate(projections):
                
                index = _block_index_(jj, block_number, projs.shape[1], 'random')
                
                proj = numpy.ascontiguousarray(projs[:,index,:])
                geom = geometries[kk]

                proj_geom = flexData.astra_proj_geom(geom, projs.shape, index = index) 
                vol_geom = flexData.astra_vol_geom(geom, volume.shape) 
            
                prj_tmp = numpy.zeros_like(proj)
                
                # Compute weights:
                if pwls & ~ student:
                    fwp_w = numpy.exp(-proj * weight_power)
                    
                else:
                    fwp_w = numpy.ones_like(proj)
                                        
                #fwp_w = scipy.ndimage.morphology.grey_erosion(fwp_w, size=(3,1,3))
                
                _backproject_block_(fwp_w, bwp_w, proj_geom, vol_geom, 'BP3D_CUDA', '+')
                
                #flex.project.backproject(fwp_w, bwp_w, geom)  
                _forwardproject_block_(prj_tmp, volume, proj_geom, vol_geom, '+')
                #flex.project.forwardproject(prj_tmp, volume, geom)
            
                if rings_t == 0:
                    prj_tmp = (proj - prj_tmp) * fwp_w / fac

                    #flex.util.display_slice(prj_tmp,dim=1, title='pre')
                    if student:
                        prj_tmp = studentst(prj_tmp, 5)
                    
                else:
                    # Add rings removal:
                    # Residual:                                
                    prj_tmp = (proj + ring[:,None,:] - prj_tmp) * fwp_w / fac
                    
                    # Update rings:
                    me = prj_tmp.mean(1) * 2
                    #rec -= me
                    ring -= (me - scipy.signal.medfilt(me, 5)) 
                    
                    ring = ring - ring.mean()
                    ring = numpy.maximum(numpy.abs(ring)-rings_t, 0) * numpy.sign(ring)
                    
                    
                _backproject_block_(prj_tmp, vol_tmp, proj_geom, vol_geom, 'BP3D_CUDA', '+')
                
                # Mean L for projection
                L_mean += (prj_tmp**2).mean() 
                
            eps = bwp_w.max() / 100    
            bwp_w[bwp_w < eps] = eps
                
            volume += vol_tmp / bwp_w
            volume[volume < 0] = 0

            #print((volume<0).sum())
                
        L.append(L_mean / block_number / len(projections))
        
        #flex.util.display_slice(vol_rec, title = 'Iter')
        flexUtil.progress_bar((ii+1)/n_iter)
        
    flexUtil.plot(numpy.array(L), semilogy=True)
    
    return ring
         
def EM(projections, volume, geometry, iterations, options = {'preview':False, 'bounds':None, 'block_number':1, 'mode':'sequential', 'l2_update': True}):
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
    
    flexUtil.progress_bar(0)
        
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
                        
        flexUtil.progress_bar((ii+1) / iterations)
        
    if options.get('l2_update'):
        flexUtil.plot(l2, semilogy = True, title = 'Resudual L2')   

def EM_tiled(projections, volume, geometries, iterations, options = {'poisson_weight': False, 'l2_update': True, 'preview':False, 'bounds':None, 'block_number':1, 'mode':'sequential', 'ctf': None}):
    """
    EM: tiled version.
    """     
    # Make sure that the volume is positive:
    if volume.max() <= 0: 
        volume *= 0
        volume += 1
    elif volume.min() < 0: volume[volume < 0] = 0

    for proj in projections:
        proj[proj < 0] = 0

    # Initialize L2:
    l2 = []
    
    # In tiled reconstructions position of the volume should be the same for each tile:
    geometries_ = []
    for geom in geometries:
        geom_ = geom.copy()
        
        # Compute average volume shift:
        geom_['vol_tra'] = numpy.mean([g['vol_tra'] for g in geometries], 0)
        geometries_.append(geom_)

    print('Em Emm Emmmm...')
    
    flexUtil.progress_bar(0)
        
    for ii in range(iterations):
        
        #l2_ = 0
        for ii, proj in enumerate(projections):
            
            geom = geometries_[ii]

            # Update volume:
            l2_ = _em_step_(proj, 1, volume, geom, options)
            
        # Preview
        if options.get('preview'):
            flexUtil.display_slice(volume, dim = 0)
            
        l2.append(l2_)
            
        flexUtil.progress_bar((ii+1) / iterations)
        
    if options.get('l2_update'):   

         plt.figure(15)
         plt.plot(l2)
         plt.title('Residual L2') 