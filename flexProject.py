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

import flexUtil
import flexData

''' * Methods * '''

def backproject(projections, volume, geometry, algorithm = 'BP3D_CUDA'):
    """
    Backproject useing standard ASTRA functionality
    """
    # Do we need to introduce ShortScan parameter?    
    
    # Initialize ASTRA geometries:
    vol_geom = flexData.astra_vol_geom(geometry, volume.shape)
    proj_geom = flexData.astra_proj_geom(geometry, projections.shape[::2])
            
    try:
    
        sin_id = astra.data3d.link('-sino', proj_geom, projections)    
        vol_id = astra.data3d.link('-vol', vol_geom, volume)
        
        projector_id = astra.create_projector('cuda3d', proj_geom, vol_geom)
    
        if algorithm == 'BP3D_CUDA':
            asex.accumulate_BP(projector_id, vol_id, sin_id)
        elif algorithm == 'FDK_CUDA':
            asex.accumulate_FDK(projector_id, vol_id, sin_id)
        else:
            raise ValueError('Unknown ASTRA algorithm type.')
        
    except:
        print("ASTRA error:", sys.exc_info())
        
    finally:
        astra.algorithm.delete(projector_id)
        astra.data3d.delete(sin_id)
        astra.data3d.delete(vol_id)
        
def forwardproject(projections, volume, geometry):
    """
    Forwardproject
    """
    
    # Initialize ASTRA geometries:
    vol_geom = flexData.astra_vol_geom(geometry, volume.shape)
    proj_geom = flexData.astra_proj_geom(geometry, projections.shape[::2])
    
    try:
        sin_id = astra.data3d.link('-sino', proj_geom, projections)
        vol_id = astra.data3d.link('-vol', vol_geom, volume)
        
        projector_id = astra.create_projector('cuda3d', proj_geom, vol_geom)
        
        asex.accumulate_FP(projector_id, vol_id, sin_id)
      
    except:
        print("ASTRA error:", sys.exc_info())
        
    finally:
        astra.algorithm.delete(projector_id)
        astra.data3d.delete(sin_id)
        astra.data3d.delete(vol_id)        

def FDK(projections, volume, geometry):
    """
    FDK
    """
    backproject(projections, volume, geometry, 'FDK_CUDA')
    
def SIRT(projections, volume, geometry, iterations, options = {'poisson_weight': False, 'l2_update': True, 'preview':True}):
    """
    SIRT
    """ 
    
    # We will use quick and dirty scaling coefficient instead of proper calculation of weights
    prj_weight = 2 / (projections.shape[1] * geometry['img_pixel'] ** 4 * volume.shape.max())
                    
    # Initialize L2:
    l2 = []

    print('Doing SIRT`y things...')
    
    misc.progress_bar(0)
        
    for ii in range(iterations):
    
        projections_ = projections.copy()
        
        forwardproject(projections_, -volume, geometry)
        
        # Take into account Poisson:
        if options['poisson_weight']:
            # Some formula representing the effect of photon starvation...
            projections_ *= numpy.sqrt(numpy.exp(-projections))
            
        projections_ *= prj_weight    
        
        backproject(projections_, volume, geometry, 'BP3D_CUDA')    

        # L2 norm:
        if options['L2_update']:
            l2.append(numpy.sqrt((projections_ ** 2).mean()))
            
        # Preview
        if options['preview']:
            flexUtil.display_slice(volume, dim = 0)
            
        misc.progress_bar((ii+1) / iterations)
        
    plt.figure(15)
    plt.plot(l2)
    plt.title('Residual L2')
        
        