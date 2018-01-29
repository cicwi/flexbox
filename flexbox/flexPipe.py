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
import gc
import os

from . import flexData
from . import flexProject
from . import flexCompute
from . import flexUtil

class Block:
    """
    A CT dataset.
    """
    
    def __init__(self, path = ''):
        """
        Initialize a dataset object.
        """
        
        self.data = []
        self.flat = []
        self.dark = []
        self.meta = []
        
        self.status = 'pending'
        self.type = 'projections'
        
        self.path = path
        
        self.todo = []
        self.done = []

    @property
    def geometry(self):
        
        if self.meta:
            return self.meta.get('geometry')
        else:
            return None
        
    def schedule(self, name, condition):
        """
        Add action to the schedule of this data block:
        """
        self.todo.append([name, condition])    
     
    def finish(self, name, condition):
        """
        Mark the action as finished
        """
        self.todo.remove([name, condition])
        self.done.append([name, condition, time.ctime()])    
                
    def isfinished(self, name, condition):
        """
        Checks if this action was applied and finished.
        """
        return not ([name, condition] in self.todo)
                
class Action:
    """
    A batch job operation applied to a nuber of datasets.
    """                
    def __init__(self, name, callback, conditions = {}, type = 'batch'):
        
        self.name = name
        self.callback = callback
        self.conditions = conditions
        self.type = type
        
        self.status = 'pending'
        self.count = 0
                       
class Pipe:
    """
    The Pipe is handling the data que and the action que. 
    Data que consists of blocks that fall down the action que until they hit the bottom or a group action.
    """     
    def __init__(self, pipe = None):
        """
        Initialize the Pipe!
        """
        # Actions and their conditions to be applied to data:
        self._action_que_ = []
        
        # Data:
        self._data_que_ = []  

        # If pipe is provided - copy it's action que!

        # Buffer for group actions:
        self._buffer_ = {}

        # Connection to other pipes:
        self._connections_ = []    

        # This one maps function names to callback funtions:
        self._callback_dictionary_ = {'scan_flexray': self._scan_flexray_, 'read_flexray': self._read_flexray_, 
        'read_all_meta': self._read_all_meta_, 'process_flex': self._process_flex_, 'crop': self._crop_,
        'merge_detectors': self._merge_detectors_, 'merge_volume': self._merge_volume_, 
        'fdk': self._fdk_,'sirt': self._sirt_, 'write_flexray': self._write_flexray_, 'cast2int':self._cast2int_, 
        'display':self._display_, 'memmap':self._memmap_, 'read_volume': self._read_volume_}
        
        # This one maps function names to condition that have to be used with them:
        self._condition_dictionary_ = {'scan_flexray': ['path'], 'read_flexray': ['sampling'], 
        'read_all_meta':[],'process_flex': [], 'crop': ['crop'],'sirt': [],
        'merge_detectors': ['memmap'], 'merge_volume':['memmap'], 'fdk': [], 'write_flexray': ['folder'], 
        'cast2int':['bounds'], 'display':[], 'memmap':['path'], 'read_volume': []}
        
        # This one maps function names to function types. There are three: batch, standby, coincident
        self._type_dictionary_ = {'scan_flexray': 'batch', 'read_flexray': 'batch', 
        'read_all_meta':'concurrent', 'process_flex': 'batch', 'crop': 'batch', 'sirt':'batch',
        'merge_detectors': 'standby', 'merge_volume':'standby', 'fdk': 'batch', 'write_flexray': 'batch', 
        'cast2int':'batch', 'display':'batch', 'memmap':'batch', 'read_volume': 'batch'}
        
    def template(self, pipe):
        """
        Copy the pipe action que to a new pipe
        """
                
        for action in pipe._action_que_:
            # Actions need to be connected to this pipe's callbacks!
            callback = self._callback_dictionary_.get(action.name)
            
            myaction = Action(action.name, callback, action.conditions, action.type)
            self._action_que_.append(myaction)
                
    def connect(self, pipe):
        """
        Connect to another pipe:copy()
        """
        self._connections_.append(pipe)

    def __del__(self):
        """
        Clean up.
        """
        self.flush()
        
    def flush(self):
        """
        Flush the data.
        """
        
        # Data:
        self._data_que_.clear()
        
        # CLear count for actions:
        for action in self._action_que_:
            action.count = 0
        
        # Buffer for the group actions:
        self._buffer_ = {}

        gc.collect()

    def clean(self):
        """
        Erase everything!
        """        
        # Actions and their conditions to be applied to data:
        self._action_que_ = []
        self._connections_ = []

        self.flush()   
        
    def actions_dictionary(self):
        """
        Get the list of recognized actions
        """
        print('List of legal actions:')
        
        for key in self._callback_dictionary_.keys():
            print(key, '   :   ', self._condition_dictionary_.get(key))

    def add_data(self, path = '', block = None):
        """
        Add a block of data to the pipe. 
        """
        
        if path != '':
            block = Block(path)
                            
        self._data_que_.append(block)
            
    def schedule(self, name, condition = {}):
        """
        Schedule an action.
        """
        # Create an action:
        callback = self._callback_dictionary_.get(name)
        if not callback: raise ValueError('Unrecognised function:', name)
        
        mincond = self._condition_dictionary_.get(name)
        if not all([(x in condition) for x in mincond]): 
            raise ValueError('Check your conditions. They should be:', mincond)
            
        act_type = self._type_dictionary_.get(name) 
        if not act_type: raise ValueError('Type of the action unknown!')
            
        action = Action(name, callback, condition, act_type)
        
        self._action_que_.append(action)
        
    def refresh_connections(self):
        
        # If connected to more pipes, run them and use their data
        if len(self._connections_) > 0:
            
            self.flush()
            
            # Get their outputs
            for pipe in self._connections_:
            
                # Run doughter pipes:
                pipe.run()    
                
                # Copy data_que:
                self._data_que_.extend(pipe._data_que_)
                
                # Reset the que status:
                for data in self._data_que_:
                    data.status = 'pending'
                    
    def refresh_schedules(self):
        """
        Refresh todo lists of data blocks
        """
        for data in self._data_que_:
            for act in self._action_que_:
                data.schedule(act.name, act.conditions)
                        
    def run(self):
        """
        Run me! Each dataset is picked from _data_que_ array and trickled down the pipe.
        """
        
        # In case connected to other pipes:
        self.refresh_connections()                
        
        self.refresh_schedules()
        
        print(' *** Starting a pipe run *** ')
                
        # While all datasets are not ready:
        while not self._is_ready_():
 
            print('------------------------')
            print('New data block in the pipline!')
            
            # Pick a data block:
            block = self._pick_data_()
            
            # Trickle the bastard down the pipe!
            for action in self._action_que_:
                
                # If this block was put on standby - stop and go to the next one.
                if (block.status == 'standby'): 
                    break
            
                # If action applies to all blocks at the same time:
                if action.type == 'coincident':
                    if action.count == 0:
                    
                        # Apply action:
                        action.count += 1
                        action.callback(block, action.conditions, action.count) 
                        block.finish(action.name, action.conditions)
                    
                # Check if action was already finished for this dataset:
                if not block.isfinished(action.name, action.conditions):
                    
                    # If the action is group action...
                    if action.type == 'standby':
                        
                        # Switch block status to standby
                        block.status = 'standby'
                        
                        # if all data is on standby:
                        if self._is_standby_(): 
                            block.status = 'pending'
                    
                    # Action counter increase:
                    action.count += 1
                                        
                    # Apply action
                    action.callback(block, action.conditions, action.count)
                            
                    # Make an end log record
                    block.finish(action.name, action.conditions)
                    
            block.status = 'ready'    
        
    def report(self):
        """
        Report on what is in the pipe.
        """
        
        print('====================================')
        print('Pipe Report')
        print('====================================')
        print('Action que:')
        for action in self._action_que_:
            
            print('Action: ', action.name, 'was called', action.count, 'times. Finished: ', self._action_ready_(action))
        
        print('====================================')
            
        print('Data que:')
        for block in self._data_que_:
            print('Data type:', block.type, '. Status: ', block.status)
        
        print('====================================')
        
    def _action_ready_(self, action):
        """
        Check if the action was applied to all datasets.
        """
        return all([block.isfinished(action.name) for block in self._data_que_])
        
    def _is_standby_(self):        
        """
        Checks if the pipe is waiting for one or more datasets.
        """
        
        return all([(block.status == 'standby') for block in self._data_que_])
        
    def _is_ready_(self):
        """
        Check if all operations are finished.
        """
        return all([(block.status == 'ready') for block in self._data_que_])
        
    def _pick_data_(self):
        """
        Pick data from the data pool
        """
        # Finds the ones pending:
        pending = [block for block in self._data_que_ if block.status == 'pending']
            
        if len(pending) == 0:                
            raise Exception('ERROR@!!!!@!! Pipe is empty...')
                        
        # Current data in the pipe:            
        return pending[0]  

    '''
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Callbacks of the pipe <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    '''       
        
    def _scan_flexray_(self, data, condition, count):
        """
        Fake operation for scanning data.
        """
        
        print('Scanning data. Output at:', condition.get('path'))
        pass   
    
    def _read_all_meta_(self, data, condition, count):
        """
        Read all meta!
        """
        
        print('Reading all metadata...')
        
        for data in self._data_que_:
            path = data.path
            samp = condition.get('sampling')
         
            if condition.get('volume'):
                data.meta = flexData.read_meta(os.path.join(path, 'meta.toml'))
                
            else:
                meta = flexData.read_log(path, 'flexray', bins = samp)
                
                # Some log files dont have theta information... they screw things up
                if (len(meta['geometry']['thetas']) == 0):
                    if data.geometry:
                        meta['geometry']['thetas'] = data.meta['geometry']['thetas']

                data.meta = meta
    
    def _read_volume_(self, data, condition, count):
        """
        """
        path = data.path
        
        samp = condition.get('sampling')
        memmap = condition.get('memmap')
        
        if not samp: samp = 1
        
        # Read volume:
        data.data = flexData.read_raw(path, 'vol', skip = samp, sample = [samp, samp], memmap = memmap)  
        
        data.meta = flexData.read_meta(os.path.join(path, 'meta.toml')) 
        
        data.type = 'volume'
            
    def _read_flexray_(self, data, condition, count):
        """
        Read data from disk.
        Possible conditions: path, samplig, memmap
        """        
        
        # Read:    
        print('Reading data...')
        
        path = data.path
        
        samp = condition.get('sampling')
        memmap = condition.get('memmap')
        
        # Read projections:    
            
        data.dark = flexData.read_raw(path, 'di', sample = [samp, samp])
        data.flat = flexData.read_raw(path, 'io', sample = [samp, samp])    
        
        data.data = flexData.read_raw(path, 'scan_', skip = samp, sample = [samp, samp], memmap = memmap)
    
        data.meta = flexData.read_log(path, 'flexray', bins = samp)   
                
        data.meta['geometry']['thetas'] = data.meta['geometry']['thetas'][::samp]

        data.data = flexData.raw2astra(data.data)    
        data.dark = flexData.raw2astra(data.dark)    
        data.flat = flexData.raw2astra(data.flat)    
        
        data.type = 'projections'
        
        # Sometimes flex files don't report theta range...
        if len(data.meta['geometry']['thetas']) == 0:
            data.meta['geometry']['thetas'] = numpy.linspace(0, 360, data.data.shape[1])
        
    def _process_flex_(self, data, condition, count):
        """
        Process data.
        """
        
        print('Processing data...')
        
        data.data -= data.dark
        data.data /= (data.flat.mean(1)[:, None, :] - data.dark)
            
        numpy.log(data.data, out = data.data)
        data.data *= -1
            
    def _merge_detectors_(self, data, condition, count):
        """
        Merge datasets one by one. 
        Condiitions: geoms, sampling, memmap
        """
        print('Merging detectors...')        
                
        # First call creates a buffer:
        if count == 1:    
        
            # Merge will kill the data after it is used. So we need to save some stuff:
            geoms = []

            for data_ in self._data_que_:
                geoms.append(data_.geometry) 
    
            # If there is no buffer set it to the first incoming dataset:                            
            tot_shape, tot_geom = flexData.tiles_shape(data.data.shape, geoms)  

            # Initialize total:
            memmap = condition.get('memmap')

            if memmap: 
                total = numpy.memmap(memmap, dtype='float32', mode='w+', shape = (tot_shape[0],tot_shape[1],tot_shape[2]))       
            else:
                total = numpy.zeros(tot_shape, dtype='float32')          
                
            self._buffer_['tot_geom'] = tot_geom
                
        else:            
            
            # Add data to existing buffer:
            total = self._buffer_['total']    
            tot_geom = self._buffer_['tot_geom']    
        
        flexCompute.append_tile(data.data, data.meta['geometry'], total, tot_geom)
        
        # Display:
        #flexUtil.display_slice(total, dim = 1,title = 'total')  
        
        self._buffer_['total'] = total
        
        if data.status == 'standby':
           # Remove data to save RAM
           self._data_que_.remove(data)
           del data 
           gc.collect()
           
        else:
           # If this is the last call:
               
           data.data = total
           data.meta['geometry'] = tot_geom

           # Replace all datasteams with the result: 
           self._data_que_ = [data,]
           self._buffer_ = {}

           gc.collect()
           
    def _merge_volume_(self, data, condition, count):
        """
        Merge datasets one by one. 
        Condiitions: geoms, sampling, memmap
        """
        print('Merging volumes...')        
        
        # First initialize the buffer with a large volume:
        if count == 1:    
               
            # Compute indexes for all volumes:    
            vol_z = []    
    
            # Get all vol z positions:
            for data_ in self._data_que_:
                if not data_.meta:
                    raise Exception('Meta data is not initialized for all data blocks! Use read_all_meta!')
                    
                vol_z.append(data_.geometry.get('vol_vrt'))
                    
            vol_z0 = min(vol_z)
            
            # Zero coordinate:
            self._buffer_['vol_z0'] = vol_z0

            # Total volume shape:
            # Compute offset relative to the volume at the bottom and indexes of the volumes:
            offset = numpy.int32(flexData.mm2pixel(max(vol_z) - min(vol_z), data.geometry))
            
            tot_shape = numpy.array(data.data.shape)
            tot_shape[0] +=  offset + 1
                    
            # Initialize total:
            memmap = condition.get('memmap')
            
            if memmap: 
                total = numpy.memmap(memmap, dtype=data.data.dtype, mode='w+', shape = (tot_shape[0],tot_shape[1],tot_shape[2]))       
            else:
                total = numpy.zeros(tot_shape, dtype=data.data.dtype)     
                
            self._buffer_['total'] = total    

        else:
            
            total = self._buffer_['total']
            vol_z0 = self._buffer_['vol_z0']

        # Index of the current dataset:
        vol_z = data.geometry.get('vol_vrt')    
        offset = numpy.int32(flexData.mm2pixel(vol_z - vol_z0, data.geometry))
        
        index = numpy.arange(0, data.data.shape[0]) + offset
                
        total[index] = numpy.max([data.data, total[index]], 0)
        
        #flexUtil.display_slice(total, dim = 1,title = 'vol merge')  

        self._buffer_['total'] = total 

        # Clear memory:
        if data.status == 'standby':
           # Remove data to save RAM
           self._data_que_.remove(data)
           del data 
           gc.collect()
           
        else:
           # If this is the last call:
               
           data.data = total

           # Replace all datasteams with the result: 
           self._data_que_ = [data,]
           self._buffer_ = {}

           gc.collect()      
    
    def _sirt_(self, data, condition, count):        
                
        shape = data.data.shape
        vol = numpy.zeros([shape[0]+40, shape[2], shape[2]], dtype = 'float32')
        
        options = {'bounds':[0, 1000], 'l2_update':False, 'block_number':100, 'index':'random'}
        flexProject.SIRT(data.data, vol, data.meta['geometry'], iterations = 5, options = options)
                
        # Replace projection data with volume data:
        data.data = vol 
        
    def _fdk_(self, data, condition, count):        
        
        vol = flexProject.init_volume(data.data)

        # Apply a small ramp to reduce filtering aretfacts:        
        ramp = condition.get('ramp')
        if ramp:
            data.data = flexCompute.apply_edge_ramp(data.data, ramp)
        
        flexProject.FDK(data.data, vol, data.meta['geometry'])

        # Replace projection data with volume data:
        data.data = vol
                
    def _crop_(self, data, condition, count):
        """
        Crop
        """
        print('Applying crop...')
        crop = condition.get('crop')
        
        if crop[0] > 0:
            data.data = data.data[crop[0]:-crop[0], :, :]

        if crop[1] > 0:
            data.data = data.data[:, crop[1]:-crop[1], :]

        if crop[2] > 0:
            data.data = data.data[:,:,crop[2]:-crop[2]]
        
    def _cast2int_(self, data, condition, count):
        """
        Cast data to int8
        """        
        print('Casting data to int...')
        
        data.data = flexData.cast2type(data.data, 'uint8', condition.get('bounds'))
                
    def _display_(self, data, condition, count):
        """
        Display some data
        """        
        dim = condition.get('dim')
        if not dim: 
            dim = 1
            
            # DIsplay single dimension:
            flexUtil.display_slice(data.data, dim = dim, title = 'Mid slice. Block #%u'%count)
                
    def _memmap_(self, data, condition, count):
        """
        Map data to disk
        """
        
        print('Mapping data to disk...')
        
        # Map to file:
        file = os.path.join(condition.get('path'), 'block_%u'%count)
        
        shape = data.data.shape
        dtype = data.data.dtype
        memmap = numpy.memmap(file, dtype=dtype, mode='w+', shape = (shape[0], shape[1], shape[2]))       
        
        memmap[:] = data.data[:]
        data.data = memmap
        
        # Clean up memory
        gc.collect()
        
    def _write_flexray_(self, data, condition, count):
        
        folder = condition.get('folder')
                        
        dim = condition.get('dim')
        if dim is None:
            dim = 0

        flexData.write_raw(os.path.join(data.path, '../', folder), 'vol', data.data, dim = dim)
        flexData.write_meta(os.path.join(data.path, '../', folder, 'meta.toml'), data.meta)  
'''
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
'''
    