#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in Jan 2018

@author: kostenko

This module allows to create a pipeline of operations. Each data unit trickles through that operation stack one by one until a group operation is encountered.
"""
# >>> Imports >>>
#import os
import numpy
#from scipy import ndimage
#from scipy import signal
import warnings
import time
import gc
import os
import sys

from . import flexData
from . import flexProject
from . import flexCompute
from . import flexUtil

# >>> Constants >>>
# These will be types of actions:
_ACTION_BATCH_ = 'batch'            # Most common type. Actions are applied one by one.
_ACTION_STANDBY_ = 'standby'          # The data block becomes "standby" until all blocks are processed by the action
_ACTION_CONCURRENT_ = 'concurrent'       # Concurrent - action is applied to all data blocks in the que.

# Status of the data:
_STATUS_PENDING_ = 'pending'
_STATUS_STANDBY_ = 'standby'
_STATUS_READY_ = 'ready'

# >>> Classes >>>

class Block:
    """
    A CT dataset.
    """
    
    def __init__(self, path = ''):
        """
        Initialize a dataset object.
        """
        
        self.data = []
        self.meta = []
        
        self.status = _STATUS_PENDING_
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
        
    def copy(self):
        
        block = Block()
        
        block.data = self.data.copy()
        block.meta = self.meta.copy()
        block.status = self.status
        block.type = self.type
        block.path = self.path
        block.todo = self.todo.copy()
        block.done = self.done.copy()
        
        return block
        
    def schedule(self, name, condition):
        """
        Add action to the schedule of this data block:
        """
        self.todo.append([name, condition])    
     
    def finish(self, name, condition):
        """
        Mark the action as finished
        """
        
        #print('Finished an action.')
        
        # Some actions may alter the data que, in that case action may be finished in a new que:
        if [name, condition] in self.todo:
            self.todo.remove([name, condition])
            self.done.append([name, condition, time.ctime()])    
        
        #print('++++++++++')
        #print('self.todo', self.todo)
        #print('self.done', self.done)
                
    def isfinished(self, name, condition):
        """
        Checks if this action was applied and finished.
        """
        return not ([name, condition] in self.todo)
        
    def flush(self):
        
        self.data = []
        
        gc.collect()
        
        # It seems that sometimes memore is released after a small delay
        time.sleep(1)
                
        print('Block flushed.')
        
    def __del__(self):
        """
        Clear the memory.
        """
        self.flush()        
                        
class Action:
    """
    A batch job operation applied to a nuber of datasets.
    """                
    def __init__(self, name, callback, type = _ACTION_BATCH_, arguments = []):
        
        self.name = name
        self.callback = callback
        self.arguments = arguments
        self.type = type
        
        self.status = _STATUS_PENDING_
        self.count = 0
                       
class Pipe:
    """
    The Pipe is handling the data que and the action que. 
    Data que consists of blocks that fall down the action que until they hit the bottom or a group action.
    """ 
    
    _ignore_warnings_ = True
    _history_ = {}
    
    def __init__(self, memmap_path = '', hostname ='', pas = '', usr = '', pipe = None):
        """
        Initialize the Pipe!
        """
        # Actions and their arguments to be applied to data:
        self._action_que_ = []
        
        # Data:
        self._data_que_ = []  

        # Current block in the data que:
        self._block_ = []
    
        # Buffer for group actions:
        self._buffer_ = {}

        # Connection to other pipes:
        self._connections_ = []    

        # Memmaps - need to delete them at the end:
        self._memmaps_ = []    
        self._memmap_path_ = memmap_path
        
        # SSH info:
        self._hostname_ = hostname
        self._pas_ = pas
        self._usr_ = usr

        # If pipe is provided - copy it's action que!
        if pipe:
            self.template(pipe)
            
    def ignore_warnings(self, ignore = True):
        """
        Switch on off the 
        """        
        self._ignore_warnings_ = ignore
        
    def template(self, pipe):
        """
        Copy the pipe action que to a new pipe
        """        
        # Copy ssh info and memmap folder::
        self._hostname_ = pipe._hostname_
        self._pas_ = pipe._pas_
        self._usr_ = pipe._usr_
        
        self._memmap_path_ = pipe._memmap_path_
        
        # Recreate action que:
        for action in pipe._action_que_:
            # Here we shouldnt copy the action callback as it carries a link to another pipe:
            callback = getattr(self, action.callback.__name__)
            
            # Create an action and add to this pipe:
            myaction = Action(action.name, callback, action.type, action.arguments)
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
        
    def flush_buffer(self):
        """
        Flush the buffer only.
        """
        self._buffer_ = {}
        gc.collect()
        
    def flush_memmaps(self):    
        """
        Remove memmaps.
        """
        for memmap in self._memmaps_:
            if os.path.isfile(memmap): os.remove(memmap)
            
        self._memmaps_ = []
         
    def flush(self):
        """
        Flush the data.
        """
        # TODO: test flushing and restarting the pipe.
        
        # Flush connected pipes:
        for pipe in self._connections_:
            pipe.flush()
            
        # Data:
        if self._data_que_:
            self._data_que_.clear()
        
        # CLear count for actions:
        for action in self._action_que_:
            action.count = 0
        
        # Buffer for the group actions:
        self.flush_buffer()
        
        # Remove my memmaps:
        self.flush_memmaps()

    def clean(self):
        """
        Erase everything!
        """        
        # Actions and their arguments to be applied to data:
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

    def _add_block_(self, block):
        '''
        Add new data block.
        '''
        
        # schedule actions to this block:
        for act in self._action_que_:
            block.schedule(act.name, act.arguments)
                        
        self._data_que_.append(block)
        
    def pull_data(self, local_path, remote_path, hostname, username, password = None):
        """
        Pull all data from the host.
        """
        flexData.ssh_get_path(hostname, username, password, local_path, remote_path)  
        
    def push_data(self, local_path, remote_path, hostname, username, password = None, cleanup = False):
        """
        Push all data to the host.
        """
        flexData.ssh_put_path(hostname, username, password, local_path, remote_path)  
        
        if cleanup:
            flexData.delete_path(local_path)
    
    def add_data(self, local_path):
        """
        Add a block of data to the pipe. 
        """
        import glob
        
        print('Adding: ', local_path)
        
        if not '*' in local_path:

            # Add a block with the given path                        
            self._add_block_(Block(local_path))
            
        else:
            # Expand and loop:
            folders = glob.glob(local_path)
            
            if len(folders) == 0:
                raise ValueError('No data found at the specified path: ' + local_path)
            
            for path_ in folders:
                if os.path.isdir(path_):
                    self._add_block_(Block(path_))  
                 
            print('Created %u data blocks.' % len(folders))     
                            
    def refresh_connections(self):
        
        # If connected to more pipes, run them and use their data
        if len(self._connections_) > 0:
            
            # Get their outputs
            for pipe in self._connections_:
            
                # Run doughter pipes:
                pipe.run()    
                
                # Copy data_que:
                self._data_que_.extend(pipe._data_que_)
                
                pipe._data_que_ = None
                
                # Reset the que status:
                for data in self._data_que_:
                    data.status = _STATUS_PENDING_
                    
    def refresh_schedules(self):
        """
        Refresh todo lists of data blocks
        """
        for data in self._data_que_:
            for act in self._action_que_:
                data.schedule(act.name, act.arguments)
   
    def run(self):
        """
        Run me! Each dataset is picked from _data_que_ array and trickled down the pipe.
        """        
        import traceback
        
        # In case connected to other pipes:
        self.refresh_connections()                
        
        #self.refresh_schedules() this causes double scheduling!
        
        print(' *** Starting a pipe run *** ')
        
        try:
            
            # Show available RAM:
            flexUtil.print_memory()    
            
            # While all datasets are not ready:
            while not self._is_ready_():
                
                # Pick a data block (flush old if needed):   
                if self._block_:
                    self._block_.flush()
                    
                self._block_ = self._pick_data_()
                
                print(' ')
                
                # Push the bastard down the pipe!
                for action in self._action_que_:
                    
                    # If this block was put on standby - stop and go to the next one.
                    if (self._block_.status == _ACTION_STANDBY_): 
                        break
                
                    # If action applies to all blocks at the same time:
                    if action.type == _ACTION_CONCURRENT_:
                        if action.count == 0:
                        
                            # Apply action:
                            action.count += 1
                            print('*Executing concurrent action: ' + action.name)
                            
                            # On/Off warnings
                            if self._ignore_warnings_:
                                warnings.filterwarnings("ignore")
                            else:
                                warnings.filterwarnings("default")
            
                            # Run action:
                            action.callback(self._block_, action.count, action.arguments) 
                            
                            # Make all blocks finish with this operation:
                            for block in self._data_que_:
                                block.finish(action.name, action.arguments)
                        
                    # Check if action was already finished for this dataset:
                    if not self._block_.isfinished(action.name, action.arguments):
                        
                        # If the action is group action...
                        if action.type == _ACTION_STANDBY_:
                            
                            # Switch block status to standby
                            self._block_.status = _STATUS_STANDBY_
                            
                            # if all data is on standby:
                            if self._is_standby_(): 
                                self._block_.status = _STATUS_PENDING_
                                
                            print('*Executing group action: ' + action.name)
                            
                        else:
                            print('*Executing batch action: ' + action.name)
                        
                        # Action counter increase:
                        action.count += 1
                        
                                      
                        #print('*** Block ***')
                        #print(block.data)
                        
                        # On/Off warnings
                        if self._ignore_warnings_:
                            warnings.filterwarnings("ignore")
                        else:
                            warnings.filterwarnings("default")
                            
                        # Apply action  
                        action.callback(self._block_, action.count, action.arguments)
                        
                        # Collect garbage:
                        gc.collect() 
                        
                        flexUtil.print_memory()
                                
                        # Make an end log record
                        self._block_.finish(action.name, action.arguments)
                        
                if self._block_.status == _STATUS_PENDING_:
                    self._block_.status = _STATUS_READY_
        
        except Exception as eerrrr: 
                        
            print("")
            print("    (×_×)     Pipe error      (×_×) ")
            print("")
            print('ERROR:', eerrrr)
            
            info = sys.exc_info()
            traceback.print_exception(*info)
            
            print('Will try to continue....')
       
            
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
        return all([block.isfinished(action.name, action.arguments) for block in self._data_que_])
        
    def _is_standby_(self):        
        """
        Checks if the pipe is waiting for one or more datasets.
        """
        
        return all([(block.status == _STATUS_STANDBY_) for block in self._data_que_])
        
    def _is_ready_(self):
        """
        Check if all operations are finished.
        """
        return all([(block.status == _STATUS_READY_) for block in self._data_que_])
        
    def _pick_data_(self):
        """
        Pick data from the data pool
        """
        
        print('Picking a new data block.')
        
        # Finds the ones pending:
        pending = [block for block in self._data_que_ if block.status == _STATUS_PENDING_]
            
        if len(pending) == 0:                
            raise Exception('ERROR@!!!!@!! Pipe is empty...')
                        
        # Current data in the pipe:            
        return pending[0]  

    def _add_action_(self, name, callback, act_type, *args):
        """
        Schedule an action.
        """
        # Add counter to the name to make it unique:
        name = '[%u]'%len(self._action_que_) + name
        
        action = Action(name, callback, act_type, args)
        
        self._action_que_.append(action)

        # If there is data in the que - schedule the new action:        
        for data in self._data_que_:
            data.schedule(action.name, action.arguments)
            
    def _buffer_to_que_(self):
       """
       Mode the buffer to the data que.
       """
        
       print('Populating data que with the buffer data.')
       # Use the first record of the data_que as a template:
       
       block = Block()
       block.type = self._data_que_[0].type
       block.path = self._data_que_[0].path
       block.todo = self._data_que_[0].todo.copy()
       block.done = self._data_que_[0].done.copy()
       
       self._data_que_.clear()
       
       for ii, buffer in enumerate(self._buffer_['tot_data']):
              
            # New block:
            new_block = block.copy()
            
            # Populate the data que with the buffer contents:
            new_block.data = buffer
            new_block.meta = flexData.create_meta(0, 0, 0)
            new_block.meta['geometry'] = self._buffer_['tot_geom'][ii]
       
            self._data_que_.append(new_block)
            
            # Clean the garbage:
            gc.collect()
       
       # Current block will be set to the first in the updated que :
       self._block_ = self._data_que_[0]
       
       print('Data que populated with the buffer content. Removing the buffer...') 
       self.flush_buffer()
       
    def _record_history_(self, key, arguments = []):
        """
        Make an internal history record.
        """
        if arguments:
            self._history_[key] = [arguments, time.ctime()]
            
        else:
            self._history_[key] = [time.ctime(),]
       
    '''
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Here is the description of actions >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    '''           
    def _arg_(self, args, index):
        """
        A little function that returns either the list element or None without raising an out of bounds error.
        """
        if len(args) > index:
            return args[index]
        else:
            return None
            
    def _read_all_meta_(self, data, count, argument):
        """
        Read all meta!
        """
        print('Reading all metadata...')
                
        for data in self._data_que_:
            path = data.path
            samp = self._arg_(argument, 0)
         
            if self._arg_(argument, 1):
                data.meta = flexData.read_meta(os.path.join(path, 'meta.toml'))
                
            else:
                
                if os.path.exists(os.path.join(path, 'metadata.toml')):
                    
                    meta = flexData.read_log(path, 'metadata', sample = samp)   
                    
                else:
                    meta = flexData.read_log(path, 'flexray', sample = samp)
                    
                data.meta = meta
    
    def read_all_meta(self, sampling = 1, volume = False):
        """
        Read all meta files. Need to call this, for instance, before merge actions.
        """
        self._add_action_('read_all_meta', self._read_all_meta_, _ACTION_CONCURRENT_, sampling, volume)
            
    def _read_volume_(self, data, count, argument):
        """
        Load the volume stack
        """
        path = data.path
        
        samp = self._arg_(argument, 0)
        memmap = self._arg_(argument, 1)
        
        if memmap:
            if not self._memmap_path_:
                raise Exception('memmap_path is not initialized in pipe!')
            
            memmap_file = os.path.join(self._memmap_path_, 'volume')
            self._memmaps_.append(memmap_file)
            
        else:
            memmap_file = None
        
        if not samp: samp = 1
        
        # Read volume:
        data.data = flexData.read_raw(path, 'vol', skip = samp, sample = [samp, samp], memmap = memmap_file)  
        
        data.meta = flexData.read_meta(os.path.join(path, 'meta.toml')) 
        
        data.type = 'volume'
        
        self._record_history_('Volume loaded.', [])
        
    def read_volume(self, sampling = 1, memmap = False):
        """
        Load the volume stack.
        """
        self._add_action_('read_volume', self._read_volume_, _ACTION_BATCH_, sampling, memmap)
                    
    def _process_flex_(self, data, count, argument):
        """
        Read data from disk.
        Possible arguments: samplig, skip, memmap
        """        
        
        # Read:    
        path = data.path
        
        samp = self._arg_(argument, 0)
        skip = self._arg_(argument, 1)
        memmap = self._arg_(argument, 2)
        
        # Keep track of memmaps:            
        if memmap:
            if not self._memmap_path_:
                raise Exception('memmap_path is not initialized in pipe!')
            
            memmap_file = os.path.join(self._memmap_path_, 'projections')
            self._memmaps_.append(memmap_file)
            
        else:
            memmap_file = None
            
        # Read and process:
        proj, meta = flexCompute.process_flex(path, samp, skip, memmap_file) 
                    
        data.data = proj
        data.meta = meta
        
        data.type = 'projections'
        
        print('Data in the pipe with shape', data.data.shape)
        
        gc.collect()
        
        self._record_history_('Standard FlexRay processing. [samp, skip]', argument[0:2])
        
    def process_flex(self, sampling = 1, skip = 1, memmap = False):
        """
        Read and process FlexRay data.
        """
        self._add_action_('process_flex', self._process_flex_, _ACTION_BATCH_, sampling, skip, memmap)
        
    def _bh_correction_(self, data, count, argument):
        """
        Correct for beam hardening.
        """
        path = self._arg_(argument, 0)
        compound = self._arg_(argument, 1)
        density = self._arg_(argument, 2)
        
        #energy, spectrum = numpy.loadtxt(os.path.join(data.path, path, 'spectrum.txt'))
        # Use toml files:
        file = os.path.join(data.path, path, 'spectrum.toml')
        if os.path.exists(file):
            spec = flexData.read_meta(file)
            
        else:
            raise Exception('File not found:' + file)
        
        data.data = flexCompute.equivalent_density(data.data, data.meta, spec['energy'], spec['spectrum'], compound = compound, density = density)
        
        self._record_history_('Beam-hardening correction. [compound, density]', [compound, density])
    
    def bh_correction(self, path, compound, density):
        """
        Correct beam hardening.
        
        path        : path to the scanner spectrum
        compound    : single material approximation compound
        density     : density of the single material
        """        
        self._add_action_('bh_correction', self._bh_correction_, _ACTION_BATCH_, path, compound, density)
    
    def _make_stl_(self, data, count, argument):
        """
        Use Marching Cubes algorithm to generate an STL file of the surface mesh after binary thresholding. 
        """
        #from stl import mesh
        
        file = self._arg_(argument, 0)
        preview = self._arg_(argument, 1)
        
        # Generate STL:
        stl_mesh = flexCompute.generate_stl(data.data, data.geometry)
        
        # Save file:
        ffile = os.path.join(data.path, file)
        print('Saving mesh at:', ffile)
        
        stl_mesh.save(ffile)
    
        # Preview:
        if preview:
            flexUtil.display_mesh(stl_mesh)
                
    def make_stl(self, file, preview = False):
        """
        Use Marching Cubes algorithm to generate an STL file of the surface mesh after binary thresholding. 
        """
        self._add_action_('make_stl', self._make_stl_, _ACTION_BATCH_, file, preview)      
               
    def _merge_detectors_(self, data, count, argument):
        """
        Merge datasets one by one. Datasets with the same source coordinates will be merged.
        arguments: geoms, sampling, memmap
        """
        
        if len(self._data_que_) == 0:
            raise Exception('The data que is empty! It has to be initialized before merge_detectors by read_all_meta')
        
        # First call creates a buffer:
        if count == 1:    
        
            # Merge will kill the data after it is used. So we need to save some stuff:
            geoms_list = []

            src_pos = []
            
            # Make a list of lists of geometries with the same source position: 
            for data_ in self._data_que_:
                
                # Source position:
                src = [data_.geometry['src_vrt'], data_.geometry['src_mag'], data_.geometry['src_hrz']]
                
                # Compare to previous source positions:
                if (src_pos is []) | (src not in src_pos):
                    src_pos.append(src)
                    geoms_list.append([data_.geometry,])
                    
                else:
                    index = src_pos.index(src)    
                    geoms_list[index].append(data_.geometry)
                
            # Check if memmap is provided:
            memmap = self._arg_(argument, 0)
            
            self._buffer_['src'] = []
            self._buffer_['tot_geom'] = []
            self._buffer_['tot_data'] = []
            
            # Compute total geometries for each of the groups:                                
            for ii, geoms in enumerate(geoms_list):
                tot_shape, tot_geom = flexData.tiles_shape(data.data.shape, geoms)  
                    
                # Create memmaps:
                if memmap: 
                    
                    if not self._memmap_path_:
                        raise Exception('memmap_path is not initialized in pipe!')
                    
                    file = os.path.join(self._memmap_path_, 'detector%u' % ii)
                                        
                    self._memmaps_.append(file)
                    total = numpy.memmap(file, dtype='float32', mode='w+', shape = (tot_shape[0],tot_shape[1],tot_shape[2]))       
                    
                else:
                    total = numpy.zeros(tot_shape, dtype='float32')          
                    
                # Dump to buffer:                    
                self._buffer_['src'].append([tot_geom['src_vrt'], tot_geom['src_mag'], tot_geom['src_hrz']])
                self._buffer_['tot_geom'].append(tot_geom)
                self._buffer_['tot_data'].append(total)
                
            print('Populated the buffers.')
            
        # Not the first call:
        #else:       
            
        # Find to which buffer this data should be added:
        src = [data.geometry['src_vrt'], data.geometry['src_mag'], data.geometry['src_hrz']]
        
        index = self._buffer_['src'].index(src)
        
        # Add data to existing buffer:
        total = self._buffer_['tot_data'][index]    
        tot_geom = self._buffer_['tot_geom'][index]
        
        # Display:
        #flexUtil.display_slice(total, dim = 1,title = 'total from buffer')  
    
        flexCompute.append_tile(data.data, data.meta['geometry'], total, tot_geom)
        
        self._buffer_['tot_data'][index] = total   
    
        # Display:
        flexUtil.display_slice(total, dim = 1,title = 'total to buffer')  
        
        if data.status == _STATUS_STANDBY_:
           pass 
           # Remove data to save RAM:
           # self._data_que_.remove(data)
           # del data 
           # gc.collect()
           
        # If this is the last call:   
        else:
            
           # Replace all datastreams with the result: 
           self._buffer_to_que_()
           
           # Clear buffer:
           #self.flush_buffer()
           
    def merge_detectors(self, memmap = False):
        """
        Merge detectors into a single image. Will produce a separate datablock for each source position. 
        memmap : path to the scratch file.
        """
        self._add_action_('merge_detectors', self._merge_detectors_, _ACTION_STANDBY_, memmap)
        
    def _find_intersection_(self, interval_a, interval_b):
        """
        Utility that computes intesection between two interwals:
        """
        
        if ( (interval_a[0] > interval_b[1]) | (interval_a[1] < interval_b[0]) ):
            return 0
        else:
            a = max(interval_a[0], interval_b[0])
            b = min(interval_a[1], interval_b[1])
            
            return b - a
                                      
    def _merge_volume_(self, data, count, argument):
        """
        Merge volume datasets one by one. 
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
                    
                z = data_.meta['geometry']['vol_tra'][0]    
                vol_z.append(z)
                    
            vol_z0 = min(vol_z)
            
            # Zero coordinate:
            self._buffer_['vol_z0'] = vol_z0

            # Total volume shape:
            # Compute offset relative to the volume at the bottom and indexes of the volumes:
            offset = numpy.int32(flexData.mm2pixel(max(vol_z) - min(vol_z), data.geometry))
            
            tot_shape = numpy.array(data.data.shape)
            tot_shape[0] +=  offset + 1
                    
            # Initialize total:
            memmap = self._arg_(argument,0)
            
            if memmap: 
                file = os.path.join(self._memmap_path_, 'volume')
                
                self._memmaps_.append(file)
                total = numpy.memmap(file, dtype=data.data.dtype, mode='w+', shape = (tot_shape[0],tot_shape[1],tot_shape[2]))       
                
            else:
                total = numpy.zeros(tot_shape, dtype=data.data.dtype)     
                
            self._buffer_['total'] = total
            
            # Find two datasets that have biggest overlap and assume it's the same for every adjuscent dataset:           
            b0 = flexData.volume_bounds(data.data.shape, self._data_que_[0].meta['geometry'])
            
            overlap = 0            
            for ii in range(1, len(self._data_que_)):
                
                b1 = flexData.volume_bounds(data.data.shape, self._data_que_[ii].meta['geometry'])
                
                new_overlap = self._find_intersection_(b0['vrt'], b1['vrt'])                
                overlap = max(overlap, new_overlap)

            # This is in volume (img_pixel):
            overlap = int(overlap / data.geometry['img_pixel'])
            
            self._buffer_['overlap'] = overlap
            
            print('Overlap between tiles is:', overlap, 'pixels')

        else:
            
            total = self._buffer_['total']
            vol_z0 = self._buffer_['vol_z0']
            overlap = self._buffer_['overlap']
            
        # Index of the current dataset:
        vol_z = data.meta['geometry']['vol_tra'][0]
        
        offset = numpy.int32(flexData.mm2pixel(vol_z - vol_z0, data.geometry))
        
        # Crop data based on the size of the overlap:
        ramp = data.data.shape[0] // 10
        
        dif = int(overlap / 2 - ramp) # to be safe....
        
        if dif > 0:
            
            print('Ramp of %u pixels is applied in volume merge. Will crop %u pixels before merge to reduce the risk of artifacts.' % (ramp, dif))
            
            data.data = data.data[dif:-dif,:,:]
            
        else:
            ramp = int(overlap / 2)
            
        print('New data shape is', data.data.shape)            
                    
        # Merge volumes with some ramp:
        index = numpy.arange(0, data.data.shape[0]) + offset
         
        jj = 0
        for ii in index[:ramp]:
            b = (jj+1) / ramp
            a = min([(total[ii].sum() != 0), 1 - b])            
            total[ii] = (data.data[jj] * b + total[ii] * a) / (a + b)
            
            jj += 1
        
        for ii in index[ramp:-ramp]:
            total[ii] = data.data[jj]
            
            jj += 1
        
        sz = index.size
        for ii in index[-ramp:]:
            b = (sz - jj) / ramp
            a = min([(total[ii].sum() != 0), 1 - b])            
            total[ii] = (data.data[jj] * b + total[ii] * a) / (a + b)
            
            jj += 1
        
            #numpy.max([data.data, total[index]], 0)
                
        #total[index] = numpy.max([data.data, total[index]], 0)
        
        flexUtil.display_slice(total, dim = 1,title = 'vol merge')  

        self._buffer_['total'] = total 

        # Clear memory:
        if data.status == _STATUS_STANDBY_:
           # Remove data to save RAM
           self._data_que_.remove(data)
           del data 
           gc.collect()
           
        else:
           # If this is the last call:
           data.data = total

           # Replace all datasteams with the result: 
           self._data_que_ = [data,]
           
           self.flush_buffer()

        gc.collect()

    def merge_volume(self, memmap):
        """
        Merge volumes vertically.
        memmap : path to the scratch file.
        """
        self._add_action_('merge_volume', self._merge_volume_, _ACTION_STANDBY_, memmap)     
        
    def _fdk_(self, data, count, argument):        

        shape = data.data.shape
        #safety = data.data.shape[0] // 10
        vol = numpy.zeros([shape[0], shape[2], shape[2]], dtype = 'float32')
        
        flexProject.FDK(data.data, vol, data.meta['geometry'])
        
        em = self._arg_(argument,0)
        sirt = self._arg_(argument,1)
        
        if em:
            flexProject.EM(data.data, vol, data.meta['geometry'], iterations = em, options = {'block_number':20})
            
        if sirt:
            flexProject.SIRT(data.data, vol, data.meta['geometry'], iterations = sirt, options = {'bounds' :[0,10], 'block_number':20})

        # Replace projection data with volume data:
        data.data = vol
        data.type = 'volume'
        
        # Try to collect the garbage:
        gc.collect()
        
        self._record_history_('FDK reconstruction [volume shape]', data.data.shape)

    def FDK(self, em = 0, sirt = 0):
        """
        Reconstruct using FDK. Use em, sirt to specify a number iterations to apply with SIRT or EM after FDK is computed.
        """
        self._add_action_('FDK', self._fdk_, _ACTION_BATCH_, em, sirt)        
            
    def _sirt_(self, data, count, argument):        
                
        shape = data.data.shape
        vol = numpy.zeros([shape[0]+10, shape[2], shape[2]], dtype = 'float32')
        
        iterations = self._arg_(argument, 0)
        block_number = self._arg_(argument, 1)
        mode = self._arg_(argument, 2)
        
        options = {'bounds':[0, 2], 'l2_update':False, 'block_number':block_number, 'mode':mode}
                
        flexProject.SIRT(data.data, vol, data.meta['geometry'], iterations = iterations, options = options)
                
        # Replace projection data with volume data:
        data.data = vol 
        data.type = 'volume'
        
        # Try to collect the garbage:
        gc.collect()
        
        self._record_history_('SIRT reconstruction [iterations, block number, volume shape]', [iterations, block_number, data.data.shape])

    def SIRT(self, iterations = 10, block_number = 40, mode = 'random'):
        """
        Run SIRT. Use block_number and mode for the subset version of SIRT.
        """
        self._add_action_('SIRT', self._sirt_, _ACTION_BATCH_, iterations, block_number, mode)
        
    def _find_rotation_(self, data, count, argument):        
        """
        Find the rotation axis:
        """
        print('Optimization of the rotation axis...')
        guess = flexCompute.optimize_rotation_center(data.data, data.meta['geometry'], centre_of_mass = False, subscale = 2)
        
        print('Old value:%0.3f' % data.meta['geometry']['axs_hrz'], 'new value: %0.3f' % guess)
        data.meta['geometry']['axs_hrz'] = guess
        
        self._record_history_('Rotation axis optimized. [offset in mm]', guess)

    def find_rotation(self):
        """
        Find the rotation center.
        """
        self._add_action_('find_rotation', self._find_rotation_, _ACTION_BATCH_)
        
    def _em_(self, data, count, argument):        
        
        shape = data.data.shape
        vol = numpy.zeros([shape[0]+10, shape[2], shape[2]], dtype = 'float32')
        
        iterations = self._arg_(argument, 0)
        block_number = self._arg_(argument, 1)
        mode = self._arg_(argument, 2)
        
        flexProject.EM(data.data, vol, data.meta['geometry'], iterations = iterations, options = {'bounds': [0, 2], 'block_number':block_number, 'mode':mode})
        
        # Replace projection data with volume data:
        data.data = vol
        data.type = 'volume'
        
        # Try to collect the garbage:
        gc.collect()
        
        self._record_history_('EM reconstruction [iterations, block number, volume shape]', [iterations, block_number, data.data.shape])
        
    def EM(self, iterations = 5, block_number = 10, mode = 'random'):
        """
        Run Expectation Maximization.
        """
        self._add_action_('EM', self._em_, _ACTION_BATCH_, iterations, block_number, mode)        
        
    def _ramp_(self, data, count, argument):
        
        width = self._arg_(argument, 0)
        dim = self._arg_(argument, 1)
        mode = self._arg_(argument, 2)
        
        if (dim == 2) & (not numpy.isscalar(width)):
            offset = (width[1] - width[0]) / 2
            data.meta['geometry']['det_hrz'] += offset * data.meta['geometry']['det_pixel']
        
        data.data = flexData.pad(data.data, dim, width, mode)
        
        self._record_history_('Ramp applied. [dim, width]', [dim, width])
        
        #data.data = flexUtil.apply_edge_ramp(data.data, width, extend)
    
    def ramp(self, width, dim, mode = 'linear'):
        """
        Apply pad and ramp to one of the dimensions.
        """
        self._add_action_('ramp', self._ramp_, _ACTION_BATCH_, width, dim, mode)
                        
    def _shape_(self, data, count, argument):
        """
        Shape the data to a given shape.
        """
        print('Applying shape...')
        
        shape = self._arg_(argument, 0)
                
        myshape = data.data.shape
        if (myshape != shape):
            
            print('Changing shape from:', myshape, 'to', shape)
            
            for dim in range(3):
                crop = shape[dim] - myshape[dim]
                if crop < 0:
                    # Crop case:
                    data.data = flexData.crop(data.data, dim, -crop, symmetric = True)
            
                elif crop > 0:
                    # Pad case:
                    data.data = flexData.pad(data.data, dim, crop, symmetric = True)
                    
        self._record_history_('Shaping applied. [shape]', shape)
                    
    def shape(self, shape):
        """
        Shape the data either by cropping or by paddig.
        """
        self._add_action_('shape', self._ramp_, _ACTION_BATCH_, shape)                    

    def _bin_(self, data, count, argument):
        """
        Crop the data.
        """
        print('Applying binning...')
        
        dim = self._arg_(argument, 0)
        
        data.data = flexData.bin(data.data, dim)
        
        self._record_history_('Binning applied. [dim]', dim)

    def bin(self, dim = None):
        """
        FBin the data in certain direction or in all at the same time.
        """
        self._add_action_('bin', self._bin_, _ACTION_BATCH_, dim)                
                    
    def _crop_(self, data, count, argument):
        """
        Crop the data.
        """
        print('Applying crop...')
        
        dim = self._arg_(argument, 0)
        width = self._arg_(argument, 1)
         
        data.data = flexData.crop(data.data, dim, width)
        
        self._record_history_('Crop applied. [dim, width]', [dim, width])

    def crop(self, dim, width):
        """
        Crop the data.
        """
        self._add_action_('crop', self._crop_, _ACTION_BATCH_, dim, width)        

    def _auto_crop_(self, data, count, argument):
        """
        Auto-crop the data.
        """
        print('Applying automatic crop...')
        
        a,b,c = flexCompute.bounding_box(data.data)                 
        data.data = data.data[a[0]:a[1], b[0]:b[1], c[0]:c[1]]
        
        self._record_history_('Auto-crop applied. [bounding box]', [a,b,c])
        
    def auto_crop(self):
        """
        Auto-crop the volume.
        """
        
        self._add_action_('auto_crop', self._auto_crop_, _ACTION_BATCH_)        

    def _marker_normalization_(self, data, count, argument):
        """
        Normalize the data using markers.
        """
        normalization_value = self._arg_(argument, 0)
    
        # Find the marker:
        a,b,c = flexCompute.find_marker(data.data, data.meta)    
        
        rho = data.data[a-1:a+1, b-1:b+1, c-1:c+1].mean()
    
        print('Marker density is: %2.2f' % rho)
        
        data.data *= (normalization_value / rho)
        
        self._record_history_('Marker based normalization. [old, new]', [rho, normalization_value])
        
    def marker_normalization(self, normalization_value = 1):
        """
        Normalize the data using markers.
        """
        
        self._add_action_('marker_normalization', self._marker_normalization_, _ACTION_BATCH_, normalization_value) 
                    
    def _cast2type_(self, data, count, argument):
        """
        Cast data to type.
        """                
        dtype = self._arg_(argument, 0)
        bounds = self._arg_(argument, 1)
        
        if dtype  is None:
            dtype = 'float16'
        
        print('Casting data to ', dtype)
        
        data.data = flexData.cast2type(data.data, dtype, bounds)
        
        self._record_history_('Change precision point. [dtype]', dtype)

    def cast2type(self, dtype, bounds):
        """
        Cast the data to the given dtype and upper and lower bounds.
        """
        
        self._add_action_('cast2type', self._cast2type_, _ACTION_BATCH_, dtype, bounds)                
                        
    def _display_(self, data, count, argument):
        """
        Display some data.
        """        
        dim = self._arg_(argument, 0)
        display_type = self._arg_(argument, 1)
        print_geom = self._arg_(argument, 2)
        
        if dim is None: 
            dim = 1
            
        # DIsplay single dimension:
        if display_type == 'projection':
            flexUtil.display_projection(data.data, dim = dim, title = 'Projection. Block #%u'%count)
                                        
        elif display_type == 'max_projection':
            flexUtil.display_max_projection(data.data, dim = dim, title = 'Max projection. Block #%u'%count)    
                                        
        elif display_type == 'slice':
            flexUtil.display_slice(data.data, dim = dim, title = 'Mid slice. Block #%u'%count)    
                                   
        if print_geom:
           print('Geometry:')
           print(data.meta['geometry'])
    
    def display(self, dim = 0, display_type = 'slice', print_geom = False):
        """
        Display data.
        """
        self._add_action_('display', self._display_, _ACTION_BATCH_, dim, display_type, print_geom)                
                
    def _memmap_(self, data, count, argument):
        """
        Map data to disk
        """
        
        print('Mapping data to disk...')
                    
        if not self._memmap_path_:
            raise Exception('memmap_path is not initialized in pipe!')
        
        memmap_file = os.path.join(self._memmap_path_, 'block_%u'%count)
        self._memmaps_.append(memmap_file)
                        
        shape = data.data.shape
        dtype = data.data.dtype
        
        memmap = numpy.memmap(memmap_file, dtype=dtype, mode='w+', shape = (shape[0], shape[1], shape[2]))       
        
        memmap[:] = data.data[:]
        data.data = memmap
        
        # Clean up memory
        gc.collect()

    def memmap(self, path):
        """
        Push data into a memmap.
        """
        self._add_action_('memmap', self._memmap_, _ACTION_BATCH_, path)                
            
    def _write_flexray_(self, data, count, argument):
        """
        Write the raw and meta files to disk.
        """
        
        folder = self._arg_(argument, 0)
        name = self._arg_(argument, 1)
        dim = self._arg_(argument, 2)
        skip = self._arg_(argument, 3)
        compress = self._arg_(argument, 4)
        
        self._record_history_('Saved to disk. [shape, dtype, zlib compression]', [data.data.shape, data.data.dtype, compress])
        
        print('Writing data at:', os.path.join(data.path, folder))
        flexData.write_raw(os.path.join(data.path, folder), name, data.data, dim = dim, skip = skip, compress = compress)
        
        print('Writing meta to:', os.path.join(data.path, folder, 'meta.toml'))
        flexData.write_meta(os.path.join(data.path, folder, 'meta.toml'), data.meta)  

    def write_flexray(self, folder, name = 'vol', dim = 0, skip = 1, compress = 'zip'):
        """
        Write the raw and meta files to disk.
        """
        self._add_action_('write_flexray', self._write_flexray_, _ACTION_BATCH_, folder, name, dim, skip, compress)
        
    def _history_to_meta_(self, data, count, argument):
        """
        Write the raw and meta files to disk.
        """
        data.meta['history'] = self._history_
        
        #flexData.write_meta(os.path.join(data.path, folder, 'meta.toml'), data.meta)  

    def history_to_meta(self):
        """
        Write the history of this pipe run into the meta record.
        """
        self._add_action_('history_to_meta', self._history_to_meta_, _ACTION_BATCH_)    
            
    def _shift_(self, data, count, argument):
        """
        Shift the data by a fixed amount.
        """
        dim = self._arg_(argument, 0)
        shift = self._arg_(argument, 1)
        
        data.data = flexCompute.translate(data.data, shift = shift, axis = dim)
        
        self._record_history_('Translation applied. [axis, shift]', dim, shift)
        
    def shift(self, dim, shift):
        """
        Shift the data along the given dimension.
        """
        self._add_action_('shift', self._shift_, _ACTION_BATCH_, dim, shift)                        
            
    def _register_volumes_(self, data, count, argument):
        """
        Register all volumes to the first one in the que.
        """
        
        print('Volume registration in progress...')
        
        # Condition of registering to the last dataset:
        last = self._arg_(argument, 0)
        
        # Compute the histogram of the first dataset:
        if count == 1:
            self._buffer_['fixed'] = data.data
                         
        else:
            # Register volumes
            T, R = flexCompute.register_volumes(self._buffer_['fixed'], data.data, subsamp = 2, use_CG = True, monochrome = False)
            
            # Resample the moving volume:
            data.data = flexCompute.affine(data.data, R, T)
            
            # We will register to the last dataset if it is mentioned in arguments:
            if last:
                self._buffer_['fixed'] = data.data    
            
        # Last call:   
        if len(self._data_que_) == count:  
            del self._buffer_['fixed']
    
    def register_volumes(self, last = False):
        """
        Register all volumes to the first one in the que. Or last ...
        """
        self._add_action_('register_volumes', self._register_volumes_, _ACTION_BATCH_, last) 
            
    def _equalize_intensity_(self, data, count, argument):
        """
        Equalize the intensity levels based on histograms.
        """
        print('Equalizing intensities...')
                
        # Compute the histogram of the first dataset:
        if count == 1:
            
            rng = flexCompute.intensity_range(data.data)
            self._buffer_['range'] = rng

            # This interefers with principal range. Use it only after!
            flexCompute.soft_threshold(data.data)
                         
        else:
             
             # Rescale intensities:
             rng_0 = self._buffer_['range']
             rng = flexCompute.intensity_range(data.data)    
                 
             print('Rescaling from [%f0.2, %f0.2] to [%f0.2, %f0.2]' % (rng[0], rng[2], rng_0[0], rng_0[2]))
             
             data.data -= (rng[0] - rng_0[0])             
             data.data *= (rng_0[2] - rng_0[0]) / (rng[2] - rng[0]) 
             
             data.data[data.data < rng_0[0]] = rng_0[0]
             
             # This interefers with principal range. Use it only after!
             flexCompute.soft_threshold(data.data)
            
        # Last call:   
        if len(self._data_que_) == count:
            del self._buffer_['range']
            
        self._record_history_('Intensity equalization applied.')

    def equalize_intensity(self):
        """
        Equalize the intensity levels based on histograms.
        """
        self._add_action_('equalize_intensity', self._equalize_intensity_, _ACTION_BATCH_)    

    def _soft_threshold_(self, data, count, argument):
        
        flexCompute.soft_threshold(data.data, self._arg_(argument, 0), self._arg_(argument,1))
        
        self._record_history_('Threshold applied. [mode, constant]', argument)

    def soft_threshold(self, mode = 'histogram', threshold = 0):
        """
        Apply binary threshold to get rid of small values.
        """
        
        self._add_action_('soft_threshold', self._soft_threshold_, _ACTION_BATCH_, mode, threshold)    
                         
    def _equalize_resolution_(self, data, count, argument):
        """
        Scale all datasets to the same pixle size. 
        """
        
        print('Equalizing pixel sizes...')
        
        # First call computes the maximum pixels and size of the volume: 
        if count == 1:
            
            # Find the biggest pixel size and volume size (can be different):
            n =len(self._data_que_)
        
            pixels = numpy.zeros(n)
            
            for ii in range(n):
                data_ = self._data_que_[ii]
                pixels[ii] = data_.meta['geometry']['img_pixel']     
         
            # Maximum pixel and maximum shape:
            pix_max = pixels.max()
            
            # For now we don't have data.data loaded for all blocks yet. SO we don't know the shape of the data.
            # Will assume that the dataset with the biggest img_pixel has the same shape as the first dataset...
            shp_max = numpy.array(data.data.shape)
            
            self._buffer_['pix_max'] = pix_max
            self._buffer_['shp_max'] = shp_max
            
            #sizes[ii] = data_.meta['geometry']['img_pixel'] * numpy.array(data_.data.shape)
            #shp_max = numpy.int32(sizes.max(0) / pix_max)
        else:
            pix_max = self._buffer_['pix_max']
            shp_max = self._buffer_['shp_max']
            
        # Downsample to that pixel size:
        fact = data.meta['geometry']['img_pixel'] / pix_max
        
        if fact != 1:
            
            print('From %uum to %uum' % (data.meta['geometry']['img_pixel']*1e3, pix_max*1e3))    
            print('fact', fact)
            
            data.data = flexCompute.scale(data.data, fact)
            data.meta['geometry']['img_pixel'] /= fact
            data.meta['geometry']['det_pixel'] /= fact

        # Equalize shape:
        myshape = data.data.shape
        if any(myshape != shp_max):
            
            print('Changing shape from:', myshape, 'to', shp_max)
            
            for dim in range(3):
                crop = shp_max[dim] - myshape[dim]
                if crop < 0:
                    # Crop case:
                    data.data = flexUtil.crop(data.data, dim, -crop, symmetric = True)
            
                elif crop > 0:
                    # Pad case:
                    data.data = flexUtil.pad(data.data, dim, crop, symmetric = True)
            
        # Report:
        mass = numpy.sum(data.data > 0) / numpy.prod(data.data.shape)    
        print('Nonzero pixels: %0.3f of the volume.' % mass)
            
        # Last call:   
        if len(self._data_que_) == count:  
            del self._buffer_['pix_max']
            del self._buffer_['shp_max']
            
        self._record_history_('Resolution equalization applied.')
            
    def equalize_resolution(self):
        """
        Scale all datasets to the same pixle size. 
        """
        self._add_action_('equalize_resolution', self._equalize_resolution_, _ACTION_BATCH_) 