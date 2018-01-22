#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in Jan 2018

@author: kostenko

This module allows to create a pipeline of operations. Each data unit trickles through that operation stack one by one until a group operation is encountered.
"""
#import os
import numpy
import time

from . import flexData

class Data:
    """
    A CT dataset.
    """
    
    def __init__(self, proj, flat, dark, meta):
        """
        Initialize a dataset object.
        """
        
        self.proj = proj
        self.flat = flat
        self.dark = dark
        self.meta = meta
        
        self.status = 'pending'
        self.memory = 'disk'
        
        self.process_log = []
        
    def log_begin(self, action, condition):
        """
        Operation begin log record.
        """       
        self.process_log.append(['started', action.__name__, condition, time.ctime()])    
    
    def log_end(self, action, condition):
        """
        Operation end log record.
        """ 
        self.process_log.append(['finished', action.__name__, condition, time.ctime()])    
               
class Pipe:
        
    def __init__(self):
        """
        Initialize the Pipe!
        """
        
        # Actions and their conditions to be applied to data:
        self._action_stack_ = []
        self._condition_stack_ = []

        # Data:
        self._datas_ = []  

        # Buffer for group operations:
        self._buffer_ = []

    def add_data(self, data):
        """
        Add a dataset object to the stack.
        """
        
        self._datas_.append(data)
        
    def scan_data(self):
        """
        A plug for a future scan data! action
        """
        self._action_stack_.append(_scan_data_)
        self._condition_stack_.append({})
                             
    def process_data(self):
        """
        Apply standard processing.
        """
        self._action_stack_.append(_process_data_)
        self._condition_stack_.append({})

    def merge_datas(self):
        """
        Apply standard processing.
        """
        self._action_stack_.append(_merge_datas_)    
        self._condition_stack_.append({'group':True})
        
    def fdk_data(self):
        """
        Reconstruct FDK.
        """
        self._action_stack_.append(_fdk_data_)    
        self._condition_stack_.append({})
        
        
        
    def run(self):
        """
        Run me! Each dataset is picked from _datas_ array and trickled down the pipe.
        """
        
        ready = False
        group_ready = False
        
        # Input for a group action if there is one
        group_input = []

        while not ready:
        
            # Pick a data:
            pending = [data for data in self._datas_ if data.status == 'pending']
            
            if len(pending) == 0:                
                print('Pipe is empty...')
                            
            # Current data in the pipe:            
            data = pending[0]  
        
            # Trickle the bastard down the pipe!
            for ii, action in enumerate(self._action_stack_):
                
                # Check if action was already finished for this dataset:
                if not data.isfinished(action.__name__):
                    
                    # Retrieve the parameters of the action
                    condition = self._condition_stack_[ii]

                    if data.status == 'waiting'
            
                    if not condition.get('group'):
                        
                        # Make a begin log record
                        data.log_begin(action, condition)
                        
                        # Apply action
                        data = action(data, condition)
            
                        # Make an end log record
                        data.log_end(action, condition)
                    
                    else:
                        
                        # Check if data is ready for a group action:
                        if group_ready:
                            
                            # Make a begin log record
                            for data_ in waiting:
                                data_.log_begin(action, condition)
                            
                            # Apply group action
                            self._datas_ = action(group_input, condition)
                            
                            group_input = []
                
                            # Make an end log record
                            for data_ in waiting:
                                data_.log_end(action, condition)    
                                
                        else:
                            # Wait for the other datasets to be processed:
                                
                            data.status == 'waiting'
                            ready = False
                        
                            group_input.append(data)
                            
                        break
                
            data.status = 'ready'
            ready = numpy.prod([data.status == 'ready' for data in self._datas_])            
        
    def _scan_data_(self, data, condition):
        """
        Fake operation for scanning data.
        """
        
        print('Scanning data. Output at:', condition.get('path'))
        pass    
    
    def _read_data_(self, data, condition):
        """
        Read data from disk.
        Possible conditions: path, samplig, disk_map
        """
        # Read:    
            
        print('Reading data...')
        
        path = condition.get('path')
        samp = condition.get('samplig')
        disk_map = condition.get('disk_map')
        
        data.dark = flexData.read_raw(path, 'di', sample = [samp, samp])
        data.flat = flexData.read_raw(path, 'io', sample = [samp, samp])    
        
        data.proj = flexData.read_raw(path, 'scan_', skip = samp, sample = [samp, samp], disk_map = disk_map)
    
        data.meta = flexData.read_log(path, 'flexray', bins = samp)   
                
        data.meta['geometry']['thetas'] = data.meta['geometry']['thetas'][::samp]

        data.proj = flexData.raw2astra(data.proj)    
        
        # Sometimes flex files don't report theta range...
        if len(data.meta['geometry']['thetas']) == 0:
            data.meta['geometry']['thetas'] = numpy.linspace(0, 360, data.proj.shape[1])
            
        return data
        
    def _process_data_(self, data, condition):
        """
        Process data.
        """
        
        print('Processing data...')
        
        data.proj -= data.dark
        data.proj /= (data.flat.mean(0) - data.dark)
            
        numpy.log(data.proj, out = data.proj)
        data.proj *= -1
        
        return data
        
    def _merge_datas_(self, data, condition):
        """
        Merge datasets
        """
        bins = 2
    proj_shape = [768, 2000, 972]
        
    # Read geometries:    
        
    geoms = []    
    for path in input_paths: 
        # meta:
        meta = flex.data.read_log(path, 'flexray', bins = bins) 
        geoms.append(meta['geometry'])
                    
    # Initialize the total data based on all geometries and a single projection stack shape:  
    tot_shape, tot_geom = flex.data.tiles_shape(proj_shape, geoms)     
    
    total = numpy.memmap('/export/scratch3/kostenko/flexbox_swap/swap.prj', dtype='float32', mode='w+', shape = (tot_shape[0],tot_shape[1],tot_shape[2]))    
    
    # Read data:
    for path in input_paths: 
        # read and process:
        proj, meta = flex.compute.process_flex(path, options = {'bin':bins, 'disk_map': None})  
        
        # Correct beam hardeinng:
        proj = flex.spectrum.equivalent_density(proj, meta['geometry'], energy, spec, compound = 'Al', density = 2.7)     
    
        flex.data.append_tile(proj, meta['geometry'], total, tot_geom)
        
        #flex.util.display_slice(total, dim = 1)
        
        # Free memory:
        del proj
        gc.collect()
        
    # TODO: fill in thetas properly
    tot_geom['thetas'] = numpy.linspace(0, 360, total.shape[1], dtype = 'float32')
    
    # Reaplce the geometry record in meta:
    meta['geometry'] = tot_geom
    
    return total, meta 
    
    
    
    
    
    def _fdk_data_(self, data, condition, prev_out):        
        pass

def write_log(filename, log):
    """
    Saves global log file.
    """
    # Will use TOML format to load/save log file
    import toml
    
    #filename = os.path.join(path, 'stack.log')
    path = os.path.dirname(filename)
    
    # Make path if does not exist:
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Save TOML to a file:
    with open(filename, 'w') as f:
        toml.dump(log, f)

def read_log(filename):
    """
    Loads global log file.
    """
    # Will use TOML format to load/save log file
    import toml
    
    #filename = os.path.join(path, 'stack.log')
        
    return toml.load(filename)
    
def init_log():
    """
    Initialize a standard global log file
    """    
    
    log = {'tile_acquire': {}, 'tile_fdk': {}}

    log['tile_acquire'] = 1

    