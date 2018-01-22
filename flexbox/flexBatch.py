#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in Jan 2018

@author: kostenko

This module will take care of batch reconstruction and, potentially, of batch acuisition of data.
"""
import os

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

    