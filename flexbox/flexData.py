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
import imageio
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
        meta (dict): description of the geometry, physical settings and comments
    '''
    dark = read_raw(path, 'di')
    
    flat = read_raw(path, 'io')
    
    proj = read_raw(path, 'scan_')
    
    meta = read_log(path, 'flexray')   
    
    return proj, flat, dark, meta
        
def read_raw(path, name, skip = 1, sample = [1, 1], x_roi = [], y_roi = [], dtype = 'float32', memmap = None, index = None):
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
        memmap (str): if provided, return a disk mapped array to save RAM
        index (array): if provided, will output an index array corresponding to succefully read files.
        
    Returns:
        numpy.array : 3D array with the first dimension representing the image index
        
    """  
        
    # Retrieve files, sorted by name
    files = _get_files_sorted_(path, name)
    
    # Create index of existing files:
    indx = numpy.array([int(re.findall(r'\d+', f)[-1]) for f in files])
    
    if indx.size > 1:
        indx //= (indx[1] - indx[0])
        indx -= indx.min()
        
        #print([ii for ii in numpy.where(indx % skip == 0)])
        reduc = numpy.where(indx % skip == 0)[0]
        
        files = [files[ii] for ii in reduc]
        indx = indx[indx % skip == 0]
    
    if len(files) == 0: raise IOError('Files not found:', os.path.join(path, name))
    
    # Read the first file:
    image = _read_tiff_(files[0], sample, x_roi, y_roi)
    sz = numpy.shape(image)
    
    file_n = len(indx)
        
    # Create a mapped array if needed:
    if memmap:
        data = numpy.memmap(memmap, dtype='float32', mode='w+', shape = (file_n, sz[0], sz[1]))
        
    else:    
        data = numpy.zeros((file_n, sz[0], sz[1]), dtype = numpy.float32)
    
    # Read all files:  
    good = []
    
    for k, ii in enumerate(indx):
        #print(ii)
        #print(files[ii])
        filename = files[k]
        
        try:
            a = _read_tiff_(filename, sample, x_roi, y_roi)
            
            # Summ RGB:    
            if a.ndim > 2:
                a = a.mean(2)
         
            #flexUtil.display_slice(a)    
            data[k, :, :] = a
            good.append(k)
        
        except:
            pass
            #a = numpy.zeros(data.shape[1:], dtype = numpy.float32)
            
        flexUtil.progress_bar((k+1) / (indx.size))

    # Get rid of the corrupted data:
    if len(good) != file_n:
        print('WARNING! %u files are CORRUPTED!'%(file_n - len(good)))
        
        indx = indx[good]
        data = data[good]

    # Output index:
    if index is not None:
        index[:] = indx
    
    print('%u files were loaded.' % file_n)

    return data    

def write_raw(path, name, data, dim = 1, skip = 1, dtype = None):
    """
    Write tiff stack.
    
    Args:
        path (str): destination path
        name (str): first part of the files name
        data (numpy.array): data to write
        dim (int): dimension along which array is separated into images
        skip (int): how many images to skip in between
        dtype (type): forse this data type       
    """
    
    print('Writing data...')
    
    # Make path if does not exist:
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Write files stack:    
    file_num = int(numpy.ceil(data.shape[dim] / skip))

    bounds = [data.min(), data.max()]
    
    for ii in range(file_num):
        
        path_name = os.path.join(path, name + '_%06u.tiff'% (ii*skip))
        
        # Extract one slice from the big array
        sl = flexUtil.anyslice(data, ii * skip, dim)
        img = data[sl]
          
        # Cast data to another type if needed
        if dtype is not None:
            img = cast2type(img, dtype, bounds)
        
        # Write it!!!
        imageio.imwrite(path_name, img)
        
        flexUtil.progress_bar((ii+1) / file_num)
        
def write_tiff(filename, image):
    """
    Write a single image.
    """ 
        
    # Make path if does not exist:
    path = os.path.dirname(filename)
    if not os.path.exists(path):
        os.makedirs(path)
    
    imageio.imwrite(filename, image)

def cast2type(array, dtype, bounds = None):
    """
    Cast from float to int or float to float rescaling values if needed.
    """
    # No? Yes? OK...
    if array.dtype == dtype:
        return array
    
    # Make sue dtype is not a string:
    dtype = numpy.dtype(dtype)
    
    # If cast to float, simply cast:
    if dtype.kind == 'f':
        return numpy.array(array, dtype)
    
    # If to integer, rescale:
    if bounds is None:
        bounds = [numpy.amin(array), numpy.amax(array)]
    
    data_max = numpy.iinfo(dtype).max
    
    array -= bounds[0]
    array *= data_max / (bounds[1] - bounds[0])
    
    array[array < 0] = 0
    array[array > data_max] = data_max
    
    array = numpy.array(array, dtype)    
    
    return array          
        
def read_log(path, name, log_type = 'flexray', bins = 1):
    """
    Read the log file and return dictionaries with parameters of the scan.
    
    Args:
        path (str): path to the files location
        name (str): common part of the files name
        log_type (bool): type of the log file
        bins: forced binning in [y, x] direction
        
    Returns:    
        geometry : src2obj, det2obj, det_pixel, thetas, det_hrz, det_vrt, det_mag, det_rot, src_hrz, src_vrt, src_mag, axs_hrz, vol_hrz, vol_tra 
        settings : physical settings - voltage, current, exposure
        description : lyrical description of the data
    """
    #print(path)
    
    if log_type != 'flexray': raise ValueError('Non-flexray log files are not supported yet. File a complaint form to the support team.')
    
    # Get dictionary to translate keywords:
    dictionary = _get_flexray_keywords_()
    
    # Read recods from the file:
    geometry, settings, description = _parse_keywords_(path, 'settings.txt', dictionary, separator = ':')
    
    # Check if all th relevant fields are there:
    _sanity_check_(geometry)
    
    # Apply Flexray-specific corrections and restructure records:
    geometry = _correct_flex_(geometry) 
    
    # Make sure that records that were not found are set to zeroes:
    #for key in geometry.keys():
    #    if type(geometry[key]) is list:
    #        for ii in range(len(geometry[key])):
    #            if geometry[key][ii] is None: 
    #                geometry[key][ii] = 0
                
    #    elif geometry[key] is None: geometry[key] = 0  

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
        
def write_astra(filename, data_shape, geometry):
    """
    Write an astra-readable projection geometry vector.
    """        
    geom = astra_proj_geom(geometry, data_shape)
    
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
    
def shape_alike(vol1, vol2):
    '''
    Make sure two arrays have the same shape:
    '''
    d_shape = numpy.array(vol2.shape)
    d_shape -= vol1.shape

    for dim in [0,1,2]:
        
        pp = d_shape[dim]
        if pp > 0:
            vol1 = pad(vol1, dim, pp, symmetric = True, mode = 'linear')
        if pp < 0:
            vol2 = pad(vol2, dim, -pp, symmetric = True, mode = 'linear')
        
    return vol1, vol2    

def ramp(array, dim, width, mode = 'linear'):
    """
    Create ramps at the ends of the array (without changing its size). 
    linear - creates linear decay of intensity
    edge - smears data in a costant manner
    zero - sets values to zeroes
    """
    
    # Left and right:
    if numpy.size(width) > 1:
        rampl = width[0]
        rampr = width[1]
    else:
        rampl = width
        rampr = width
        
    if array.shape[dim] < (rampl + rampr):
        return array
    
    # Index of the left ramp:
    left_sl = flexUtil.anyslice(array, slice(0, rampl), dim)
    right_sl = flexUtil.anyslice(array, slice(-rampr, None), dim)
    
    if mode == 'zero':    
        if rampl > 0:
            array[left_sl] *= 0
            
        if rampr > 0:    
            array[right_sl] *= 0
            
    elif (mode == 'edge'):
        # Set everything to the edge value:
        if rampl > 0:
            array[left_sl] *= 0
            flexUtil.add_dim(array[left_sl], array[flexUtil.anyslice(array, rampl, dim)])            
            
        if rampr > 0:    
            array[right_sl] *= 0
            flexUtil.add_dim(array[right_sl], array[flexUtil.anyslice(array, -rampr, dim)])            
    
    elif mode == 'linear':
        # Set to edge and multiply by a ramp:
        
        if rampl > 0:            
            # Replace values using add_dim:
            array[left_sl] *= 0
            flexUtil.add_dim(array[left_sl], array[flexUtil.anyslice(array, rampl, dim)])            
            
            flexUtil.mult_dim(array[left_sl], numpy.linspace(0, 1, rampl))        
            
        if rampr > 0:    
            # Replace values using add_dim:
            array[right_sl] *= 0
            flexUtil.add_dim(array[right_sl], array[flexUtil.anyslice(array, -rampr, dim)])            

            flexUtil.mult_dim(array[right_sl], numpy.linspace(1, 0, rampr))                    
        
    else:
        raise(mode, '- unknown mode! Use linear, edge or zero.')
        
    return array

#def pad(array, dim, width, symmetric = False, mode = 'edge'):
def pad(array, dim, width, mode = 'edge'):
    """
    Pad an array along a given dimension.
    
    numpy.pad seems to be very memory hungry! Don't use it for large arrays.
    """
    print('Padding data...')
    
    if numpy.size(width) > 1:
        padl = width[0]
        padr = width[1]
    else:
        padl = width
        padr = width
    
    # Original shape:
    
    sz1 = numpy.array(array.shape)    
    
    sz1[dim] += padl + padr
    
    # Initialize bigger array:
    new = numpy.zeros(sz1, dtype = array.dtype)    
    
    sl = flexUtil.anyslice(new, slice(padl,-padr), dim)
    
    new[sl] = array
    
    return ramp(new, dim, width, mode)
 
def bin(array):
    """
    Simple binning of the data:
    """           
    # First apply division by 8:
    if (array.dtype.kind == 'i') | (array.dtype.kind == 'u'):    
        array //= 8
    else:
        array /= 8
    
    # Try to avoid memory overflow here:
    for ii in range(array.shape[0]):
        sl = flexUtil.anyslice(array, ii, 0)
        
        x = array[sl]
        array[sl][:-1:2,:] += x[1::2,:]
        array[sl][:,:-1:2] += x[:,1::2]
        
    for ii in range(array.shape[2]):
        sl = flexUtil.anyslice(array, ii, 2)
        
        array[sl][:-1:2,:-1:2] += array[sl][1::2,:-1:2]    
        
    return array[:-1:2, :-1:2, :-1:2]
    
def crop(array, dim, width, symmetric = False, geometry = None):
    """
    Crop an array along the given dimension.
    """
    if numpy.size(width) > 1:
        widthl = int(width[0])
        widthr = int(width[1])
        
    else:
        if symmetric:
            widthl = int(width) // 2
            widthr = int(width) - widthl 
        else:
            widthl = 0
            widthr = int(width)
   
    # Geometry shifts:
    h = 0
    v = 0
        
    if dim == 0:
        v = (widthl - widthr)
        array = array[widthl:-widthr, :,:]
        
    elif dim == 1:
        h = (widthl - widthr)
        array = array[:,widthl:-widthr,:]
        
    elif dim == 2:
        h = (widthl - widthr)
        array = array[:,:,widthl:-widthr]   
    
    if geometry: shift_geometry(geometry, h/2, v/2)
    #if geometry: shift_geometry(geometry, -h/2, -v/2)
    
    return array
    
def raw2astra(array):
    """
    Convert a given numpy array (sorted: index, hor, vert) to ASTRA-compatible projections stack
    """    
    # Don't apply ascontignuousarray on memmaps!    
    array = numpy.transpose(array, [1,0,2])
    
    #Flip:
    array = array[::-1]    
        
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

def shift_geometry(geometry, hrz, vrt):
    """
    Apply geometry shift in pixels.
    """    
    hrz = hrz * geometry['det_pixel']
    vrt = vrt * geometry['det_pixel']
    
    geometry['det_hrz'] += hrz
    geometry['det_vrt'] += vrt
    
    # Here we are computing magnification without taking into account vol_tra[1], det_mag
    m = (geometry['src2obj'] + geometry['det2obj']) / geometry['src2obj']
    geometry['vol_tra'][2] += hrz / m
    geometry['vol_tra'][0] += vrt / m    
    
def astra_vol_geom(geometry, vol_shape, slice_first = None, slice_last = None):
    '''
    Initialize volume geometry.        
    '''
    # Shape and size (mm) of the volume
    vol_shape = numpy.array(vol_shape)
    
    #voxel = numpy.array([sample[0], sample[1], sample[1]]) * geometry['det_pixel'] / mag
    #mag = (geometry['det2obj'] + geometry['src2obj']) / geometry['src2obj']
    
    # Use 'img_pixel' to override the voxel size:
    sample =  geometry.get('anisotropy')   
    voxel = numpy.array([sample[0], sample[1], sample[2]]) * geometry['img_pixel']

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
        
    vol_geom = astra.creators.create_vol_geom(shape[1], shape[2], shape[0], 
              -size[2]/2, size[2]/2, -size[1]/2, size[1]/2, 
              -size[0]/2 + offset, size[0]/2 + offset)
        
    return vol_geom   

def astra_proj_geom(geometry, data_shape, index = None):
    """
    Generate the vector that describes positions of the source and detector.
    """
    # Basic geometry:
    det_count_x = data_shape[2]
    det_count_z = data_shape[0]
    theta_count = data_shape[1]

    sample =  geometry.get('sample')   
    det_pixel = geometry['det_pixel'] * numpy.array(sample)
    
    src2obj = geometry['src2obj']
    det2obj = geometry['det2obj']

    # Check if _thetas_ are defined explicitly:
    if geometry.get('_thetas_') is not None:
        thetas = geometry['_thetas_'] / 180 * numpy.pi
        
        if len(thetas) != theta_count: 
            raise IndexError('Length of the _thetas_ array doesn`t match withthe number of projections: %u v.s. %u' % (len(thetas), theta_count))
    else:
        
        thetas = numpy.linspace(geometry.get('theta_min'), geometry.get('theta_max'),theta_count, dtype = 'float32') / 180 * numpy.pi

    # Inintialize ASTRA projection geometry to import vector from it
    if (index is not None):
        
        thetas = thetas[index]
       
    proj_geom = astra.creators.create_proj_geom('cone', det_pixel[1], det_pixel[0], det_count_z, det_count_x, thetas, src2obj, det2obj)
    
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
        R = transforms3d.euler.euler2mat(geometry['vol_rot'][0], geometry['vol_rot'][1], geometry['vol_rot'][2], 'rzyx')

        # Apply transformation:
        det_axis_hrz[:] = numpy.dot(det_axis_hrz, R)
        det_axis_vrt[:] = numpy.dot(det_axis_vrt, R)
        src_vect[:] = numpy.dot(src_vect,R)
        det_vect[:] = numpy.dot(det_vect,R)            
                
        # Add translation:
        vect_norm = numpy.sqrt((det_axis_vrt ** 2).sum())

        # Take into account that the center of rotation should be in the center of reconstruction volume:        
        # T = numpy.array([geometry['vol_mag'] * vect_norm / det_pixel[1], geometry['vol_hrz'] * vect_norm / det_pixel[1], geometry['vol_vrt'] * vect_norm / det_pixel[0]])    
        T = numpy.array([geometry['vol_tra'][1] * vect_norm / det_pixel[1], geometry['vol_tra'][2] * vect_norm / det_pixel[1], geometry['vol_tra'][0] * vect_norm / det_pixel[0]])    
        
        src_vect[:] -= numpy.dot(T, R)           
        det_vect[:] -= numpy.dot(T, R)
        
    #print('vol_tra', geometry['vol_tra'])
    #print('det_pixel', det_pixel)
    #print('vect_norm', vect_norm)
    #print('T', T)
    
        
    proj_geom['Vectors'] = vectors    
    
    return proj_geom   

def create_geometry(src2obj, det2obj, det_pixel, theta_range):
    """
    Initialize an empty geometry record.
    """
    
    # Create an empty dictionary:
    geometry = {'det_pixel':det_pixel, 'det_hrz':0., 'det_vrt':0., 'det_mag':0., 
    'src_hrz':0., 'src_vrt':0., 'src_mag':0., 'axs_hrz':0., 'det_rot':0., 'anisotropy':[1,1,1],
    'vol_rot':[0. ,0. ,0.], 'vol_hrz':0., 'vol_tra':[0., 0., 0.], 'vol_mag':0., 'sample':[1,1,1],
    'src2obj': src2obj, 'det2obj':det2obj, 'unit':'millimetre', 'type':'flex', 'binning': 1}
    
    geometry['src2det'] = geometry.get('src2obj') + geometry.get('det2obj')
    
    # Add img_pixel:
    if src2obj != 0:    
        m = (src2obj + det2obj) / src2obj
        geometry['img_pixel'] = det_pixel / m
    else:
        geometry['img_pixel'] = 0

    # Generate thetas explicitly:
    geometry['theta_max'] = theta_range[1]
    geometry['theta_min'] = theta_range[0]
    #geometry['theta_num'] = theta_count

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
    
    # Update volume center:
    #geometry['vol_vrt'] = (geometry['det_vrt'] * geometry['src2obj'] + geometry['src_vrt'] * geometry['det2obj']) / geometry.get('src2det')
    #geometry['vol_hrz'] = (geometry['det_hrz'] + geometry['src_hrz']) / 2
    geometry['vol_tra'][0] = (geometry['det_vrt'] * geometry['src2obj'] + geometry['src_vrt'] * geometry['det2obj']) / geometry.get('src2det')
    #geometry['vol_tra'][2] = (geometry['det_hrz'] + geometry['src_hrz']) / 2
    geometry['vol_tra'][2] = geometry['axs_hrz']

    return new_shape, geometry
                 
def _read_tiff_(file, sample = [1, 1], x_roi = [], y_roi = []):
    """
    Read a single image.
    """
        
    # SOmetimes files dont have an extension. Fix it!
    if os.path.splitext(file)[1] == '':
        #im = imageio.imread(file, format = 'tif', offset = 0)
        im = imageio.imread(file, format = 'tif')
    else:
        #im = imageio.imread(file, offset = 0)
        im = imageio.imread(file)
        
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
                    'last angle':'theta_max',
                    'start angle':'theta_min',
                    
                    'binning value':'binning',
                    'roi (ltrb)':'roi'}
                    
    settings =     {'tube voltage':'voltage',
                    'tube power':'power',
                    'number of averages':'averages',
                    'imaging mode':'mode',
                    'scan duration':'duration',
                    'filter':'filter',
                    
                    'exposure time (ms)':'exposure'}

    description =  {'sample name' : 'comments',
                    'comment' : 'name',                    

                    'date':'date'}
                    
    return [geometry, settings, description]                
   
def _sanity_check_(records):
    
    minimum_set = ['img_pixel', 'src2det', 'src2obj']

    for word in minimum_set:
        if word not in records: raise ValueError('Missing records in the meta data. Something wrong with the log file?')
        
        if type(records[word]) != float: raise ValueError('Wrong records in the meta data. Something wrong with the log file?', word, records[word])

def _correct_flex_(records):   
    """
    Apply some Flexray specific corrections to the geometry record.
    """
    #binning = records['binning']
    #print(records.get('src2det'))
    #print(records.get('src2obj'))
    #print(records.get('src2det') - records.get('src2obj') )
    #print('***')
        
    records['det2obj'] = records.get('src2det') - records.get('src2obj')    
    records['img_pixel'] = records.get('img_pixel') * _parse_unit_('[um]') 
    
    records['det_hrz'] += 24    
    records['src_vrt'] -= 5

    # Rotation axis:
    records['axs_hrz'] -= 0.5
    
    # Compute the center of the detector:
    roi = numpy.int32(records.get('roi').split(sep=','))
    records['roi'] = roi.tolist()

    centre = [(roi[0] + roi[2]) // 2 - 971, (roi[1] + roi[3]) // 2 - 767]
    
    # Take into account the ROI of the detector:
    records['det_vrt'] -= centre[1] / records.get('binning') * records['det_pixel']
    records['det_hrz'] -= centre[0] / records.get('binning') * records['det_pixel']
    
    vol_center = (records['det_vrt'] * records['src2obj'] + records['src_vrt'] * records['det2obj']) / records.get('src2det')
    #records['vol_vrt'] = vol_center
    records['vol_tra'][0] = vol_center
    
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
    geometry = create_geometry(0, 0, 0, [0, 360])

    settings = {}
    description = {}

    # Keep record of names to avoid reading doubled-entries:
    names = []    

    # Loop to read the file record by record:
    with open(log_file, 'r') as logfile:
        for line in logfile:
            name, var = line.partition(separator)[::2]
            name = name.strip().lower()
            
            # Dont mind empty lines:
            if re.search('[a-zA-Z]', name):
            
                # Check if name occured before:
                if (name in names) & ('date' not in name): 
                    
                    print('Double occurance of:', name, '. Stopping.')
                    break
                names.append(name)
                
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
    
    
    

