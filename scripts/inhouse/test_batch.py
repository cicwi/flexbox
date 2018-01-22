#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 14:53:57 2018

@author: kostenko
"""
#%%
import flexBatch

block_path = '/export/scratch2/kostenko/archive/Natrualis/pitje/femur/high_res/femur_batch/block.log'
log = flexBatch.read_log(block_path)

flexBatch.write_log(block_path, log)

#%%

def write_log(log):
    flexBatch.write_log(log['path'], log)

    
class log_file:
    def __init__(self, path, type_):
        self._path = path      
        
        # REad existing log from file:
        self._log = flexBatch.read_log(block_path)        
        
        # Create an empty log:
        if self._log is None:
            self['type'] = type_
        
    def __getitem__(self, key):       
        
        # If item doesn't exsit - create one!
        val = self._log.get(key)
        
        if val is None:
            self[key] = []

        return val
                                
    def __setitem__(key, val):        
        # Update item and save on disk
        self._log[key] = val
        flexBatch.write_log(self._path, self._log)
        
    def __len__(self):
        return len(self._log)

def execute_action(log, key, func):
    
    action = log[key] 

    # If action is not done yet, do it!
    if action != None:
        if action['status'] != "ready":
            action['status'] = "executing"

            # execute this:
            func(log)
                        
            action['time'] = time.asctime()            
            action['status'] != "ready" 

            log[key] = action  

def extract_paths(par):
    '''
    Extract all paths mentioned inside a given field of a log file:
    '''
    return par[[k for k in par.keys() if 'path' in k]]
              
def block_scan(par):
    
    print('Block scan executing...')
    
    # Should it run a single script file or each block gets one script?
    
    path_keys = [k for k in par.keys() if 'path' in k]
    for path in par[path_keys]:
        # scan in this path:
            
        # Make path if does not exist:
        if not os.path.exists(path):
            os.makedirs(path)
        
        execute_action(log_file(path, 'stack_log'), 'stack_scan', stack_scan)  
        
def block_fdk(par):
    
    print('Block FDK executing...')
    
    path_keys = [k for k in par.keys() if 'path' in k]
    for path in par[path_keys]:
        # fdk in this path:
            
        # Merge!
        execute_action(log_file(path, 'stack_log'), 'stack_merge', stack_merge)  
        
        # FDK!
        execute_action(log_file(path, 'stack_log'), 'stack_fdk', stack_fdk)  
  
    
def block_merge(log, par):
    
    print('Block merge executing...')
    
    path_keys = [k for k in par.keys() if 'path' in k]
    for path in par[path_keys]:
        
        # Merge!
        execute_action(log_file(path, 'stack_log'), 'stack_merge', stack_merge())  
        
       
def _pass_(log, par):
    pass

def execute_action_stack(log, key):
    
    par = log[key] 
    func = command_dictionary[key]

    # If action is not done yet, do it!
    if par != None:
        if par['status'] != "ready":
            par['status'] = "executing"

            # execute this:
            func(log)
                        
            par['time'] = time.asctime()            
            par['status'] != "ready" 

            log[key] = par 

def stack_fdk(log):
    print('Stack FDK executing...')
    
    # find paths:
    paths = extract_paths(log['stack_scan'])
    
    # Should it run a single script file or each block gets one script?
    execute_action_stack(log, 'stack_merge', flexBatch.stack_merge, paths)
    execute_action_stack(log, 'stack_scan', flexBatch.merge_stacks, paths)
    
def stack_merge(log):
    print('Stack merge executing...')
    
    # find paths:
    paths = extract_paths(log['stack_scan'])
    
    # Should it run a single script file or each block gets one script?
    if len(paths) < 1:
        print('ERROR: No output paths found in stack_scan record!')
        
    else:
        
    
def stack_scan(log):
    print('Stack scan executing...')
    
    # Should it run a single script file or each block gets one script?
    # TODO: initiate actual scan here.
    
    # Now we just add folder names to the log...
    
    path = log['path']
    
    counter = 1
    
    for path in os.listdir(path):
        if not os.path.isfile(path): 
            log['path_%u' % counter] = path
            counter += 1
           
    #...
    
def stack_process()    
    
#%% Execute blocks:

#log = flexBatch.read_log(block_path)
#command_dictionary = {'block_scan':block_scan, 'block_fdk':block_fdk, 'block_merge':block_merge, 'title':_pass_, 'comment':_pass_, 'type':_pass_, 'path':_pass_}

stack_path = '/export/scratch2/kostenko/archive/Natrualis/pitje/femur/high_res/femur_batch/block_1/stack.log'
log = flexBatch.read_log(stack_path)
log['path'] = stack_path

command_dictionary = {'stack_scan':stack_scan, 'stack_merge':stack_merge, 'stack_fdk':stack_fdk, 'title':_pass_, 'comment':_pass_, 'type':_pass_, 'path':_pass_}

for key in log.keys():
    
   if key not in command_dictionary:
       print('ERROR: Unrecognized record in block log file:', key)
       
   else:
       execute_action_stack(log, key, command_dictionary[key], log):
       

       
#%% Execute stacks:



