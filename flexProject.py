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

import flexData

''' * Methods * '''

def backproject_additive(projections, volume, proj_geom, vol_geom, algorithm = 'BP3D_CUDA'):
    """
    Backproject without erasing old values
    """
    # Do we need to introduce ShortScan parameter?    
            
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
        astra.data3d.delete(sin_id)
        astra.data3d.delete(vol_id)
        astra.projector.delete(projector_id)

def backproject(projections, volume, geometry, algorithm = 'BP3D_CUDA'):
    """
    Backproject useing standard ASTRA functionality
    """
    # Do we need to introduce ShortScan parameter?    
    
    # Initialize ASTRA geometries:
    vol_geom = get_vol_geom(volume.shape, geometry)
    proj_geom = get_proj_geom(geometry, proj.shape[::2])
            
    try:
        sin_id = astra.data3d.link('-sino', proj_geom, projections)
        
        vol_id = astra.data3d.link('-vol', vol_geom, volume)
        
        cfg = astra.astra_dict(algorithm)
        cfg['ReconstructionDataId'] = vol_id
        cfg['ProjectionDataId'] = sin_id
        
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id, 1)
      
    except:
        print("ASTRA error:", sys.exc_info())
        
    finally:
        astra.algorithm.delete(alg_id)
        astra.data3d.delete(sin_id)
        astra.data3d.delete(vol_id)
        
def forwardproject(projections, volume, geometry):
    """
    Forwardproject
    """
    
    # Initialize ASTRA geometries:
    vol_geom = get_vol_geom(volume.shape, geometry)
    proj_geom = get_proj_geom(geometry, proj.shape[::2])
    
    try:
        sin_id = astra.data3d.link('-sino', proj_geom, projections)
        vol_id = astra.data3d.link('-vol', vol_geom, volume)
        
        cfg = astra.astra_dict('FP3D_CUDA')
        
        cfg['VolumeDataId'] = vol_id
        cfg['ProjectionDataId'] = sin_id
        
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id, 1)
      
    except:
        print("ASTRA error:", sys.exc_info())
        
    finally:
        astra.algorithm.delete(alg_id)
        astra.data3d.delete(sin_id)
        astra.data3d.delete(vol_id)        

def FDK(projections, volume, geometry):
    """
    FDK
    """
    backproject(projections, volume, geometry, 'FDK_CUDA')
    
def SIRT(projections, volume, geometry, iterations)
    """
    SIRT
    """ 
    
    # We will use quick and dirty scaling coefficient instead of proper calculation of weights
    prj_weight = 2 / (self.projections[0].data.length * self.volume.meta.geometry.img_pixel[0]**4 * self.volume.data.shape.max())
                    
    # Initialize L2:
    l2 = []

    volume = self.volume
    
    print('Doing SIRT`y things...')
    
    misc.progress_bar(0)        
    for ii in range(iterations):
    
        # Here loops of blocks and projections stack can be organized diffrently
        # I will implement them in the easyest way now.
        _l2 = 0
        # Loop over different projection stacks:
        for proj in self.projections:
            
            slice_shape = proj.data.slice_shape
                        
            # Loop over blocks:
            for proj_data in proj.data:
                
                # Make sure that our projection data pool is not updated by the forward projection
                # update the buffer only to keep residual in it
                proj.data._read_only = True
                
                # Geometry of the block:
                proj_geom = proj.meta.geometry.get_proj_geom(slice_shape, blocks = True)
                
                # Forward project:    
                self._forwardproject(proj_data, proj_geom, volume, -1)    
                
                # Take into account Poisson:
                if self.options['poisson_weight']:
                    # Some formula representing the effect of photon starvation...
                    proj_data *= numpy.sqrt(numpy.exp(-proj_data)) * prj_weight 

                else:
                    # Apply weights to forward projection residual:
                    proj_data *= prj_weight
                    
                # L2 norm:
                if self.options['L2_update']:
                    #print('L2',numpy.sqrt((proj_data ** 2).mean()))
                    _l2 += numpy.sqrt((proj_data ** 2).mean()) / proj.data.block_number 
                
                self._backproject(proj_data, proj_geom, volume)
                
            proj.data._read_only = False
            
        l2.append(_l2)
        
        # Preview
        if self.options['preview']:
            self.volume.display.slice(dim = 0)
            
        misc.progress_bar((ii+1) / iterations)
        
        plt.figure(15)
        plt.plot(l2)
        plt.title('Residual L2')
        
        self.volume.meta.history.add_record('Reconstruction generated using SIRT.', iterations)
    #return backproject(projections, volume, proj_geom, vol_geom, 'FDK_CUDA')
        