#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: kostenko
Created on Wed Nov 2017

This module will contain read / write routines to convert FlexRay scanner data into ASTRA compatible data

We can now read/write:
    image files (tiff stacks)
    log files from Flex ray
    toml geometry files
"""

''' * Imports * '''

import numpy
import os
import re
import astra 
import transforms3d
import transforms3d.euler

from . import flexUtil

''' * Methods * '''

def read_flexray(path):
    '''
    Read raw projecitions, dark and flat-field, scan parameters from FlexRay
    
    Args:
        path (str): path to flexray data.
        
    Returns:
        proj (numpy.array): projections stack
        flat (numpy.array): reference flat field images
        dark (numpy.array): dark field images   
        geom (dict): description of the deometry 
        phys (dict): description of the physical settings
        lyric (dict): comments
    '''
    dark = read_raw(path, 'di')
    
    flat = read_raw(path, 'io')
    
    proj = read_raw(path, 'scan_')
    
    geom, phys, lyric = read_log(path, 'flexray')   
    
    return proj, flat, dark, geom, phys, lyric
        
def read_raw(path, name, skip = 1, sample = [1, 1], x_roi = [], y_roi = [], dtype = 'float32', disk_map = None):
    """
    Read tiff files stack and return numpy array.
    
    Args:
        path (str): path to the files location
        name (str): common part of the files name
        skip (int): read every so many files
        sample (int): sampling factor in x/y direction
        x_roi ([x0, x1]): horizontal range
        y_roi ([y0, y1]): vertical range
        dtype (str or numpy.dtype): data type to return
        disk_map (str): if provided, return a disk mapped array to save RAM
        
    Returns:
        numpy.array : 3D array with the first dimension representing the image index
        
    """    
    # Retrieve files, sorted by name
    files = _get_files_sorted_(path, name)
    
    # Read the first file:
    image = _read_tiff_(files[0], sample, x_roi, y_roi)
    sz = numpy.shape(image)
    
    file_n = len(files) // skip
    
    # Create a mapped array if needed:
    if disk_map:
        data = numpy.memmap(disk_map, dtype='float32', mode='w+', shape = (file_n, sz[0], sz[1]))
        
    else:    
        data = numpy.zeros((file_n, sz[0], sz[1]), dtype = numpy.float32)
    
    # Read all files:
    for ii in range(file_n):
        
        filename = files[ii*skip]
        try:
            a = _read_tiff_(filename, sample, x_roi, y_roi)
        except:
            print('WARNING! FILE IS CORRUPTED. CREATING A BLANK IMAGE: ', filename)
            a = numpy.zeros(data.shape[1:], dtype = numpy.float32)
            
        if a.ndim > 2:
          a = a.mean(2)
          
        data[ii, :, :] = a

        flexUtil.progress_bar((ii+1) / (numpy.shape(files)[0] // skip))

    print('%u files were loaded.' % file_n)

    return data    

def write_raw(path, name, data, dim = 1, dtype = None):
    """
    Write tiff stack.
    
    Args:
        path (str): destination path
        name (str): first part of the files name
        data (numpy.array): data to write
        dim (int): dimension along which array is separated into images
        dtype (type): forse this data type       
    """
    
    import imageio
    
    # Make path if does not exist:
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Write files stack:    
    file_num = data.shape[dim]

    bounds = [data.min(), data.max()]
    
    for ii in range(file_num):
        
        path_name = os.path.join(path, name + '_%06u.tiff'%ii)
        
        # Extract one slice from the big array
        sl = flexUtil.anyslice(data, ii, dim)
        img = data[sl]
          
        # Cast data to another type if needed
        if dtype is not None:
            img = flexUtil.cast2type(img, dtype, bounds)
        
        # Write it!!!
        imageio.imwrite(path_name, img)
        
        flexUtil.progress_bar((ii+1) / file_num)
        
def write_tiff(filename, image):
    """
    Write a single image.
    """ 
    import imageio
    
    # Make path if does not exist:
    path = os.path.dirname(filename)
    if not os.path.exists(path):
        os.makedirs(path)
    
    imageio.imwrite(filename, image)           
        
def read_log(path, name, log_type = 'flexray', bins = 1):
    """
    Read the log file and return dictionaries with parameters of the scan.
    
    Args:
        path (str): path to the files location
        name (str): common part of the files name
        log_type (bool): type of the log file
        bins: forced binning in [y, x] direction
        
    Returns:    
        geometry : src2obj, det2obj, det_pixel, thetas, det_hrz, det_vrt, det_mag, det_rot, src_hrz, src_vrt, src_mag, axs_hrz, vol_hrz, vol_vrt, vol_mag, vol_rot
        settings : physical settings - voltage, current, exposure
        description : lyrical description of the data
    """
    
    if log_type != 'flexray': raise ValueError('Non-flexray log files are not supported yet. File a complaint form to the support team.')
    
    # Get dictionary to translate keywords:
    dictionary = _get_flexray_keywords_()
    
    # Read recods from the file:
    geometry, settings, description = _parse_keywords_(path, 'settings.txt', dictionary, separator = ':')
    
    # Apply Flexray-specific corrections and restructure records:
    geometry = _correct_flex_(geometry) 
    
    # Generate thetas explicitly:
    if geometry.get('theta_count') is None:
        ValueError('theta_count was not foud in the log file! thetas will not be initialized.')
    else:    
        geometry['thetas'] = numpy.linspace(geometry.get('first_angle'), geometry.get('last_angle'), geometry.get('theta_count'), dtype = 'float32')
    
    # Make sure that records that were not found are set to zeroes:
    for key in geometry.keys():
        if type(geometry[key]) is list:
            for ii in range(len(geometry[key])):
                if geometry[key][ii] is None: 
                    geometry[key][ii] = 0
                
        elif geometry[key] is None: geometry[key] = 0  

    # Apply binning if needed:
    geometry['det_pixel'] *= bins
    geometry['img_pixel'] *= bins
        
    # Create a meta record:
    meta = {'geometry':geometry, 'settings':settings, 'description':description}    
    
    return meta

def write_meta(filename, meta):
    """
    Read
    
    Args:
        
    Returns:    
    """
    
    import toml
    
    # Make path if does not exist:
    path = os.path.dirname(filename)
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Save TOML to a file:
    with open(filename, 'w') as f:
        toml.dump(meta, f)
        
def write_astra(filename, data_shape, meta):
    """
    Write an astra-readable projection geometry vector.
    """        
    geom = astra_proj_geom(meta['geometry'], data_shape)
    
    # Make path if does not exist:
    path = os.path.dirname(filename)
    if not os.path.exists(path):
        os.makedirs(path)
    
    numpy.savetxt(filename, geom['Vectors']) 
    
def read_meta(file_path):
    """
    Args:
        
    Returns:
    """  
    import toml
    
    # Read string from a file:
    #with open(file_path, 'r') as myfile:
    #    string = myfile.read()#.replace('\n', '')
    
    # Parse TOML string:
    return toml.load(file_path)
    
def raw2astra(array):
    """
    Convert a given numpy array (sorted: index, hor, vert) to ASTRA-compatible projections stack
    """    
    # Don't apply ascontignuousarray on memmaps!    
    array = numpy.transpose(array, [1,0,2])
    array = numpy.flip(array, 0)
    
    return array

def pixel2mm(value, geometry):
    """
    Convert pixels to millimetres by multiplying the value by img_pixel 
    """
    m = (geometry['src2obj'] + geometry['det2obj']) / geometry['src2obj']
    img_pixel = geometry['det_pixel'] / m

    return value * img_pixel
      
def mm2pixel(value, geometry):
    """
    Convert millimetres to pixels by dividing the value by img_pixel 
    """
    m = (geometry['src2obj'] + geometry['det2obj']) / geometry['src2obj']
    img_pixel = geometry['det_pixel'] / m

    return value / img_pixel
    
def astra_vol_geom(geometry, vol_shape, slice_first = None, slice_last = None, sample = [1, 1]):
    '''
    Initialize volume geometry.        
    '''
    # Shape and size (mm) of the volume
    vol_shape = numpy.array(vol_shape)
    
    mag = (geometry['det2obj'] + geometry['src2obj']) / geometry['src2obj']

    voxel = numpy.array([sample[0], sample[1], sample[1]]) * geometry['det_pixel'] / mag

    size = vol_shape * voxel

    if (slice_first is not None) & (slice_last is not None):
        # Generate volume geometry for one chunk of data:
                   
        length = vol_shape[0]
        
        # Compute offset from the centre:
        centre = (length - 1) / 2
        offset = (slice_first + slice_last) / 2 - centre
        offset = offset * voxel[0]
        
        shape = [slice_last - slice_first + 1, vol_shape[1], vol_shape[2]]
        size = shape * voxel[0]

    else:
        shape = vol_shape
        offset = 0     
        
    vol_geom = astra.create_vol_geom(shape[1], shape[2], shape[0], 
              -size[2]/2, size[2]/2, -size[1]/2, size[1]/2, 
              -size[0]/2 + offset, size[0]/2 + offset)
        
    return vol_geom   

def astra_proj_geom(geometry, data_shape, index = None, sample = [1, 1]):
    """
    Generate the vector that describes positions of the source and detector.
    """
    # Basic geometry:
    det_count_x = data_shape[1]
    det_count_z = data_shape[0]

    det_pixel = geometry['det_pixel'] * numpy.array(sample)
    
    src2obj = geometry['src2obj']
    det2obj = geometry['det2obj']

    thetas = geometry['thetas'] / 180 * numpy.pi

    # Inintialize ASTRA projection geometry to import vector from it
    if (index is not None):
        
        thetas = thetas[index]
       
    proj_geom = astra.create_proj_geom('cone', det_pixel[1], det_pixel[0], det_count_z, det_count_x, thetas, src2obj, det2obj)
    
    proj_geom = astra.functions.geom_2vec(proj_geom)
      
    vectors = proj_geom['Vectors']
    
    # Modify vector and apply it to astra projection geometry:
    for ii in range(0, vectors.shape[0]):

        # Define vectors:
        src_vect = vectors[ii, 0:3]    
        det_vect = vectors[ii, 3:6]    
        det_axis_hrz = vectors[ii, 6:9]          
        det_axis_vrt = vectors[ii, 9:12]

        #Precalculate vector perpendicular to the detector plane:
        det_normal = numpy.cross(det_axis_hrz, det_axis_vrt)
        det_normal = det_normal / numpy.sqrt(numpy.dot(det_normal, det_normal))
        
        # Translations relative to the detecotor plane:
    
        #Detector shift (V):
        det_vect += geometry['det_vrt'] * det_axis_vrt / det_pixel[0]

        #Detector shift (H):
        det_vect += geometry['det_hrz'] * det_axis_hrz / det_pixel[1]

        #Detector shift (M):
        det_vect += geometry['det_mag'] * det_normal /  det_pixel[1]

        #Source shift (V):
        src_vect += geometry['src_vrt'] * det_axis_vrt / det_pixel[0]

        #Source shift (H):
        src_vect += geometry['src_hrz'] * det_axis_hrz / det_pixel[1]

        #Source shift (M):
        src_vect += geometry['src_mag'] * det_normal / det_pixel[1]

        # Rotation axis shift:
        det_vect -= geometry['axs_hrz'] * det_axis_hrz  / det_pixel[1]
        src_vect -= geometry['axs_hrz'] * det_axis_hrz  / det_pixel[1]

        # Rotation relative to the detector plane:
        # Compute rotation matrix
    
        T = transforms3d.axangles.axangle2mat(det_normal, geometry['det_rot'])
        
        det_axis_hrz[:] = numpy.dot(T.T, det_axis_hrz)
        det_axis_vrt[:] = numpy.dot(T, det_axis_vrt)
    
        # Global transformation:
        # Rotation matrix based on Euler angles:
        R = transforms3d.euler.euler2mat(geometry['vol_rot'][0], geometry['vol_rot'][1], geometry['vol_rot'][2], 'szxy')

        # Apply transformation:
        det_axis_hrz[:] = numpy.dot(R, det_axis_hrz)
        det_axis_vrt[:] = numpy.dot(R, det_axis_vrt)
        src_vect[:] = numpy.dot(R, src_vect)
        det_vect[:] = numpy.dot(R, det_vect)            
                
        # Add translation:
        vect_norm = det_axis_vrt[2]
        
        T = numpy.array([geometry['vol_mag'] * vect_norm / det_pixel[1], geometry['vol_hrz'] * vect_norm / det_pixel[1], geometry['vol_vrt'] * vect_norm / det_pixel[0]])    
        src_vect[:] -= T            
        det_vect[:] -= T
    
    proj_geom['Vectors'] = vectors    
    
    return proj_geom   

def create_geometry(src2obj, det2obj, det_pixel, theta_range, theta_count):
    """
    Initialize an empty geometry record.
    """
    
    # Create an empty dictionary:
    geometry = {'det_pixel':det_pixel, 'det_hrz':0., 'det_vrt':0., 'det_mag':0., 
    'src_hrz':0., 'src_vrt':0., 'src_mag':0., 'axs_hrz':0., 'det_rot':0., 
    'vol_rot':[0. ,0. ,0.], 'vol_hrz':0., 'vol_vrt':0., 'vol_mag':0.,
    'src2obj': src2obj, 'det2obj':det2obj, 'unit':'millimetre', 'type':'flex', 'binning': 1}
    
    # Add img_pixel:
    if src2obj != 0:    
        m = (src2obj + det2obj) / src2obj
        geometry['img_pixel'] = det_pixel / m

    # Generate thetas explicitly:
    geometry['thetas'] = numpy.linspace(theta_range[0], theta_range[1], theta_count, dtype = 'float32') 

    return geometry 
        
def detector_size(shape, geometry):
    '''
    Get the size of detector in mm.
    '''       
    return geometry['det_pixel'] * numpy.array(shape)

def detector_bounds(shape, geometry):
    '''
    Get the boundaries of the detector in mm
    '''   
    bounds = {}

    xmin = geometry['det_hrz'] - geometry['det_pixel'] * shape[2] / 2
    xmax = geometry['det_hrz'] + geometry['det_pixel'] * shape[2] / 2

    ymin = geometry['det_vrt'] - geometry['det_pixel'] * shape[0] / 2
    ymax = geometry['det_vrt'] + geometry['det_pixel'] * shape[0] / 2

    bounds['hrz'] = [xmin, xmax]
    bounds['vrt'] = [ymin, ymax]
    
    return bounds 

    
def tiles_shape(shape, geometry_list):
    """
    Compute the size of the stiched dataset.
    Args:
        shape: shape of a single projection stack.
        geometry_list: list of geometries.
        
    """
    # Phisical detector size:
    min_x, min_y = numpy.inf, numpy.inf
    max_x, max_y = -numpy.inf, -numpy.inf
    
    det_pixel = geometry_list[0]['det_pixel']
    
    # Find out the size required for the final dataset
    for geo in geometry_list:
        
        bounds = detector_bounds(shape, geo)
        
        min_x = min([min_x, bounds['hrz'][0]])
        min_y = min([min_y, bounds['vrt'][0]])
        max_x = max([max_x, bounds['hrz'][1]])
        max_y = max([max_y, bounds['vrt'][1]])
        
    # Big slice:
    new_shape = numpy.array([(max_y - min_y) / det_pixel, shape[1], (max_x - min_x) / det_pixel])                     
    new_shape = numpy.int32(numpy.ceil(new_shape))  
    
    # Copy one of the geometry records and sett the correct translation:
    geometry = geometry_list[0].copy()
    
    geometry['det_hrz'] = (max_x + min_x) / 2
    geometry['det_vrt'] = (max_y + min_y) / 2
    
    return new_shape, geometry
    
def append_tile(data, geom, tot_data, tot_geom):
    """
    Append a tile to a larger dataset.
    Args:
        
        data: projection stack
        geom: geometry descritption
        tot_data: output array
        tot_geom: output geometry
        
    """ 
    
    import scipy.ndimage.interpolation as interp

    print('Stitching a tile...')               
    
    # Assuming all projections have equal number of angles and same pixel sizes
    total_shape = tot_data.shape[::2]
    det_shape = data.shape[::2]
    
    total_size = detector_size(total_shape, tot_geom)
    det_size = detector_size(det_shape, geom)
                    
    # Offset from the left top corner:
    x0 = tot_geom['det_hrz']
    y0 = tot_geom['det_vrt']
    
    x = geom['det_hrz']
    y = geom['det_vrt']
        
    x_offset = ((x - x0) + total_size[1] / 2 - det_size[1] / 2) / geom['det_pixel']
    y_offset = ((y - y0) + total_size[0] / 2 - det_size[0] / 2) / geom['det_pixel']
    
    # Pad image to get the same size as the total_slice:        
    pad_x = tot_data.shape[2] - data.shape[2]
    pad_y = tot_data.shape[0] - data.shape[0]  
    
    flexUtil.progress_bar(0)    
    
    for ii in range(tot_data.shape[1]):   
        
        # Pad to match sizes:
        proj = numpy.pad(data[:, ii, :], ((0, pad_y), (0, pad_x)), mode = 'constant')  
        
        # Apply shift:
        proj = interp.shift(proj, [y_offset, x_offset], order = 1)
        
        # Add:
        tot_data[:, ii, :] = numpy.max((proj, tot_data[:, ii, :]), 0)
        
        flexUtil.progress_bar((ii+1) / tot_data.shape[1])
             
def _read_tiff_(file, sample = [1, 1], x_roi = [], y_roi = []):
    """
    Read a single image.
    """
    import imageio
    
    im = imageio.imread(file, offset = 0)
    
    # TODO: Use kwags offset  and size to apply roi!
    
    if (y_roi != []):
        im = im[y_roi[0]:y_roi[1], :]
    if (x_roi != []):
        im = im[:, x_roi[0]:x_roi[1]]

    if sample != 1:
        im = im[::sample[0], ::sample[1]]
    
    return im

def _get_flexray_keywords_():                  
    """
    Create dictionary needed to read FlexRay log file.
    """
    # Dictionary that describes the Flexray file:        
    geometry =     {'voxel size':'img_pixel',
                    'sod':'src2obj',
                    'sdd':'src2det',
                    
                    'ver_tube':'src_vrt',
                    'ver_det':'det_vrt',
                    'tra_det':'det_hrz',
                    'tra_obj':'axs_hrz',
                    'tra_tube':'src_hrz',
                    
                    '# projections':'theta_count',
                    'last angle':'last_angle',
                    'start angle':'first_angle',
                    
                    'binning value':'binning',
                    'roi (ltrb)':'roi'}
                    
    settings =     {'tube voltage':'voltage',
                    'tube power':'power',
                    'number of averages':'averages',
                    'imaging mode':'mode',
                    'scan duration':'duration',
                    'filter':'filter',
                    
                    'exposure time (ms)':'exposure'}

    description =  {'Sample name' : 'comments',
                    'Comment' : 'name',                    

                    'date':'date'}
                    
    return [geometry, settings, description]                
   
def _correct_flex_(records):   
    """
    Apply some Flexray specific corrections to the geometry record.
    """
    #binning = records['binning']
    
    records['det2obj'] = records.get('src2det') - records.get('src2obj')    
    records['img_pixel'] = records.get('img_pixel') * _parse_unit_('[um]') 
    
    records['det_hrz'] += 24    
    records['src_vrt'] -= 5

    # Rotation axis:
    records['axs_hrz'] -= 0.5
    
    # Compute the center of the detector:
    roi = numpy.int32(records.get('roi').split(sep=','))
    centre = [(roi[0] + roi[2]) // 2 - 971, (roi[1] + roi[3]) // 2 - 767]
    
    # Take into account the ROI of the detector:
    records['det_vrt'] -= centre[1] / records.get('binning')
    records['det_hrz'] -= centre[0] / records.get('binning')
    
    vol_center = (records['det_vrt'] + records['src_vrt']) / 2
    records['vol_vrt'] = vol_center
    
    maginfication = (records['det2obj'] + records['src2obj']) / records['src2obj']

    records['det_pixel'] = records['img_pixel'] * maginfication  

    records['type'] = 'flexray'          
    records['unit'] = 'millimetre'  
        
    return records
    
def _parse_keywords_(path, file_mask, dictionary, separator = ':'):
    '''
    Parse a text file using the keywords dictionary and create a dictionary with values
    '''
    
    # Try to find the log file in the selected path and file_mask
    log_file = [x for x in os.listdir(path) if (os.path.isfile(os.path.join(path, x)) and file_mask in os.path.join(path, x))]

    # Check if there is one file:
    if len(log_file) == 0:
        raise FileNotFoundError('Log file not found in path: ' + path)
        
    if len(log_file) > 1:
        print('Found several log files. Currently using: ' + log_file[0])
        log_file = os.path.join(path, log_file[0])
    else:
        log_file = os.path.join(path, log_file[0])

    # Create an empty geometry dictionary:
    geometry = create_geometry(0, 0, 0, [0, 360], 0)

    settings = {}
    description = {}

    # Loop to read the file record by record:
    with open(log_file, 'r') as logfile:
        for line in logfile:
            name, var = line.partition(separator)[::2]
            name = name.strip().lower()
            
            # If name contains one of the keys (names can contain other stuff like units):
            _interpret_record_(name, var, dictionary[0], geometry)
            _interpret_record_(name, var, dictionary[1], settings)
            _interpret_record_(name, var, dictionary[2], description)    
               
    return geometry, settings, description 
    
def _interpret_record_(name, var, keywords, output):
    """
    If the record matches one of the keywords, output it's value.
    """
    geom_key = [keywords[key] for key in keywords.keys() if key in name]

    # Collect record values:
    if geom_key != []:
        
        # Look for unit description in the name:
        factor = _parse_unit_(name)

        #if geom_key[0] in output:
        #    print('*: ', geom_key[0])
        #    print('**: ', output)
        #    
        #    print('WARNING! Geometry record found twice in the log file!')
            
        # If needed to separate the var and save the number of save the whole string:   
        try:
            output[geom_key[0]] = float(var.split()[0]) * factor

        except:
            output[geom_key[0]] = var

def _parse_unit_(string):
    '''
    Look for units in the string and return a factor that converts this unit to Si.
    '''
    
    # Look at the inside a braket:
    if '[' in string:
        string = string[string.index('[')+1:string.index(']')]
                    
    elif '(' in string:
        string = string[string.index('(')+1:string.index(')')]

    else:
        return 1 
        
    # Here is what we are looking for:
    units_dictionary = {'um':0.001, 'mm':1, 'cm':10.0, 'm':1e3, 'rad':1, 'deg':numpy.pi / 180.0, 'ms':1, 's':1e3, 'us':0.001, 'kev':1, 'mev':1e3, 'ev':0.001,
                        'kv':1, 'mv':1e3, 'v':0.001, 'ua':1, 'ma':1e3, 'a':1e6, 'line':1}    
                        
    factor = [units_dictionary[key] for key in units_dictionary.keys() if key == string]
    
    if factor == []: factor = 1
    else: factor = factor[0]

    return factor    
             
def _get_files_sorted_(path, name):
    """
    Sort file entries using the natural (human) sorting
    """
    # Get the files
    files = os.listdir(path)
    
    # Get the files that are alike and sort:
    files = [os.path.join(path,x) for x in files if (name in x)]

    # Keys
    keys = [int(re.findall('\d+', f)[-1]) for f in files]

    # Sort files using keys:
    files = [f for (k, f) in sorted(zip(keys, files))]

    return files 
    
    
    

