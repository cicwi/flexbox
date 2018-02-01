#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 17:34:43 2018

@author: kostenko

Test of the flexPipe.
"""
     
#%% Advanced setup (nested pipes):
import flexbox as flex
   
# Pipes for processing projections and volume: 
pipe_vol = flex.pipe.Pipe()
pipe_proj = flex.pipe.Pipe()

# Shedule actions for pipe_proj:
pipe_proj.schedule('read_flexray', {'sampling': 2})
pipe_proj.schedule('read_all_meta', {'sampling': 2})
pipe_proj.schedule('process_flex')
pipe_proj.schedule('merge_detectors', {'memmap': '/export/scratch3/kostenko/flexbox_swap/det_swap.prj'})
pipe_proj.schedule('crop', {'crop':[7, 0, 0]})  # This crop is applied to trim shifts introduced by merge_detectors - they are bad for filter-artefacts
pipe_proj.schedule('display')
pipe_proj.schedule('find_rotation')
pipe_proj.schedule('fdk', {'ramp':15})
pipe_proj.schedule('cast2int', {'bounds':[0, 0.2]})
pipe_proj.schedule('crop', {'crop':[0, 150, 150]})  # Crop to save some disk space
pipe_proj.schedule('write_flexray', {'folder':'fdk', 'dim':0})
pipe_proj.schedule('display')

# Shedule actions for pipe_vol:
pipe_vol.schedule('merge_volume', {'memmap': '/export/scratch3/kostenko/flexbox_swap/vol_swap.prj'})
pipe_vol.schedule('display')
pipe_vol.schedule('write_flexray', {'folder':'total', 'dim':0})

# Connect big pipe_vol to many pipes:
for ii in range(6):
    
    # Children pipes (copy of pipe_proj):
    pipe = flex.pipe.Pipe(pipe_proj)
        
    block_path = '/export/scratch2/kostenko/archive/Natrualis/pitje/femur/high_res/femur_batch/block_%u/'%(ii+1)
    pipe.add_data(path = block_path + 'stack_1')
    pipe.add_data(path = block_path + 'stack_2')
    
    # Connect each pipe_a to pipe_vol:
    pipe_vol.connect(pipe)    

# RUN!!!!
pipe_vol.run()
pipe_vol.report()
pipe_vol.flush()
