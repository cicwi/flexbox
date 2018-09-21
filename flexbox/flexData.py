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
import warnings
import shutil

import paramiko 
import errno
import stat
import flexbox as flex

from . import flexUtil

''' * Methods * '''

GEOM_SIMPLE = 'simple'
GEOM_STATIC_OFFSETS = 'static_offsets'
GEOM_LINEAR_OFFSETS = 'linear_offsets'

def read_flexray(path, sample = 1, skip = 1, memmap = None, index = None):
    '''
    Read raw projecitions, dark and flat-field, scan parameters from FlexRay
    
    Args:
        path   (str): path to flexray data.
        skip   (int): read every ## image
        sample (int): keep every ## x ## pixel
        memmap (str): path to memmap file if needed
        index(array): index of the files that could be loaded
        
    Returns:
        proj (numpy.array): projections stack
        flat (numpy.array): reference flat field images
        dark (numpy.array): dark field images   
        meta (dict): description of the geometry, physical settings and comments
    '''
    dark = read_raw(path, 'di', skip, sample)
    flat = read_raw(path, 'io', skip, sample)
    
    # Read the raw data
    proj = read_raw(path, 'scan_', skip, sample, [], [], 'float32', memmap, index)
    
    # Try to retrieve metadata:
    if os.path.exists(os.path.join(path, 'metadata.toml')):
        
        meta = read_log(path, 'metadata', sample)   
        
    else:
        
        meta = read_log(path, 'flexray', sample)   
    
    return proj, flat, dark, meta

def read_log(path, log_type = 'flexray', sample = 1):
    """
    Read the log file and return dictionaries with parameters of the scan.
    
    Args:
        path (str): path to the files location
        log_type (bool): type of the log file
        bins: forced binning in [y, x] direction
        
    Returns:    
        geometry : src2obj, det2obj, det_pixel, thetas, det_hrz, det_vrt, det_mag, det_rot, src_hrz, src_vrt, src_mag, axs_hrz, vol_hrz, vol_tra 
        settings : physical settings - voltage, current, exposure
        description : lyrical description of the data
    """
    
    if log_type == 'flexray': 
        
        # Read file:
        records = _file_to_dictionary_(path, 'settings.txt', separator = ':')
        
        # Translate:
        meta = _flexray_log_translate_(records)
        
    elif log_type == 'metadata': 
        
        # Read file:
        records = _file_to_dictionary_(path, 'metadata.toml', separator = '=')
        
        # Combine all records together:
        #records_ = {}
        #records_.update(records['Settings'])
        #records_.update(records['Geometry'])
        #records_.update(records['Miscellaneous'])
                
        # Translate:
        meta = _metadata_translate_(records)
        
    else:
        raise ValueError('Unknown log_type: ' + log_type)
      
    # Convert units to standard:    
    _convert_units_(meta)    
        
    # Check if all th relevant fields are there:
    _sanity_check_(meta)
    
    # Apply external binning if needed:
    meta['geometry']['det_pixel'] *= sample
    meta['geometry']['img_pixel'] *= sample
        
    return meta
        
def read_raw(path, name, skip = 1, sample = 1, x_roi = [], y_roi = [], dtype = 'float32', memmap = None, index = None):
    """
    Read tiff files stack and return a numpy array.
    
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
    
    if (indx.size > 1):
        
        # This doesnt work with MEDIPIX files because they dont have an index at the end of the file name
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
        warnings.warn('%u files are CORRUPTED!'%(file_n - len(good)))
                
        indx = indx[good]
        data = data[good]

    # Output index:
    if index is not None:
        index[:] = indx
    
    print('%u files were loaded.' % file_n)

    return data    

def write_raw(path, name, data, dim = 1, skip = 1, dtype = None, compress = None):
    """
    Write a tiff stack.
    
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
        
        path_name = os.path.join(path, name + '_%06u'% (ii*skip))
        
        # Extract one slice from the big array
        sl = flexUtil.anyslice(data, ii * skip, dim)
        img = data[sl]
          
        # Cast data to another type if needed
        if dtype is not None:
            img = cast2type(img, dtype, bounds)
        
        # Write it!!!
        if (compress == 'zip'):
            write_tiff(path_name + '.tiff', img, 1)
            
        elif (not compress):
            write_tiff(path_name + '.tiff', img, 0)
            
        elif compress == 'jp2':  
            write_tiff(path_name + '.jp2', img, 1)
            
            '''
            To enable JPEG 2000 support, you need to build and install the OpenJPEG library, version 2.0.0 or higher, before building the Python Imaging Library.
            conda install -c conda-forge openjpeg
            '''
            
        else:
            raise ValueError('Unknown compression!')
                
        flexUtil.progress_bar((ii+1) / file_num)
                
def write_tiff(filename, image, compress = 0):
    """
    Write a single tiff image.
    """ 
        
    # Make path if does not exist:
    #path = os.path.dirname(filename)
    #if not os.path.exists(path):
        #os.makedirs(path)
    
    # imageio.imwrite(filename, image) - this can't pass compression parameter (see imageio.help('tif'))
    with imageio.get_writer(filename) as w:
        w.append_data(image, {'compress': compress})
    
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

def create_geometry(src2obj, det2obj, det_pixel, theta_range = [0, 360], type = 'simple'):
    """
    Initialize the geometry record with a basic geometry records.
    """
    if src2obj != 0:
        img_pixel = det_pixel / ((src2obj + det2obj) / src2obj)
        
    else:
        img_pixel = 0
        
    # Create a geometry dictionary:
    geometry = {'type':type, 'unit':'millimetre', 'det_pixel':det_pixel, 
                'src2obj': src2obj, 'det2obj':det2obj, 'theta_min': theta_range[0], 'theta_max': theta_range[1],
                
                'img_pixel':img_pixel, 'vol_sample':[1, 1, 1], 'proj_sample':[1, 1, 1], 
                'vol_rot':[0.,0.,0.], 'vol_tra':[0.,0.,0.]
                }
    
    # If type is not 'simple', populate with additional records:
    if type == GEOM_STATIC_OFFSETS:
        geometry['src_vrt'] = 0.
        geometry['src_hrz'] = 0.
        geometry['src_mag'] = 0. # This value should most of the times be zero since the SOD and SDD distances are known
        
        geometry['det_vrt'] = 0.
        geometry['det_hrz'] = 0.
        geometry['det_mag'] = 0. # same here
        geometry['det_rot'] = 0.
        
        geometry['axs_hrz'] = 0.
        geometry['axs_mag'] = 0. # same here
        
        
    if type == GEOM_LINEAR_OFFSETS:
        zz = numpy.zeros(2, dtype = 'float')
        geometry['src_vrt'] = zz
        geometry['src_hrz'] = zz
        geometry['src_mag'] = zz # This value should most of the times be zero since the SOD and SDD distances are known
        
        geometry['det_vrt'] = zz
        geometry['det_hrz'] = zz
        geometry['det_mag'] = zz # same here
        geometry['det_rot'] = zz
        
        geometry['axs_hrz'] = zz
        geometry['axs_mag'] = zz # same here 
        
    return geometry

def create_meta(src2obj, det2obj, det_pixel, theta_range = [0, 360], type = 'simple'):
    """
    Initialize the meta record with a basic geometry records.
    """
    
    # Settings and description:
    settings = {'voltage': 0,
                'power': 0,
                'averages': 0,
                'mode':'n/a',
                'filter':'n/a',
                'exposure':0}
    
    description = { 'name':'n/a',
                    'comments':'n/a',
                    'date':'n/a',
                    'duration':'n/a'
                    }
        
    # Geometry:
    geometry = create_geometry(src2obj, det2obj, det_pixel, theta_range, type)     
                            
    return {'geometry':geometry, 'settings':settings, 'description':description}         

        
def _convert_units_(meta):
    '''
    Converts a meta record to standard units (mm).
    '''
    
    unit = _parse_unit_(meta['geometry']['unit'])
    
    meta['geometry']['det_pixel'] *= unit
    meta['geometry']['src2obj'] *= unit
    meta['geometry']['det2obj'] *= unit
    
    meta['geometry']['src_vrt'] *= unit
    meta['geometry']['src_hrz'] *= unit
    meta['geometry']['src_mag'] *= unit
    
    meta['geometry']['det_vrt'] *= unit
    meta['geometry']['det_hrz'] *= unit
    meta['geometry']['det_mag'] *= unit
    
    meta['geometry']['axs_hrz'] *= unit
    meta['geometry']['axs_mag'] *= unit
    
    meta['geometry']['unit'] = 'millimetre'
    
    meta['geometry']['img_pixel'] *= unit
    meta['geometry']['vol_tra'] *= unit

def _flexray_log_translate_(records):                  
    """
    Translate records parsed from the Flex-Ray log file (scan settings.txt) to the meta object.
    
    """
    # If the file was not found:
    if records is None: return None
    
    # Initialize empty meta record:
    meta = create_meta(0, 0, 0, [0, 360], GEOM_STATIC_OFFSETS)
    
    # Dictionary that describes the Flexray log record:        
    geom_dict =     {'img_pixel':'voxel size',
                     'det_pixel':'binned pixel size',
                    
                    'src2obj':'sod',
                    'src2det':'sdd',
                    
                    'src_vrt':'ver_tube',
                    'src_hrz':'tra_tube',
                    
                    'det_vrt':'ver_det',
                    'det_hrz':'tra_det',                    
                    
                    'axs_hrz':'tra_obj',
                    
                    'theta_max':'last angle',
                    'theta_min':'start angle',
                    
                    'roi':'roi (ltrb)'}
                    
    sett_dict =     {'voltage':'tube voltage',
                    'power':'tube power',
                    'averages':'number of averages',
                    'mode':'imaging mode',
                    'filter':'filter',
                    
                    'exposure':'exposure time (ms)',
                    
                    'binning':'binning value',
                    
                    'dark_avrg' : '# offset images',
                    'pre_flat':'# pre flat fields',
                    'post_flat':'# post flat fields'}
    
    descr_dict =    {'duration':'scan duration',
                    'name':'sample name',
                    'comments' : 'comment', 
                    
                    'samp_size':'sample size',
                    'owner':'sample owner',

                    'date':'date'}
    
    # Translate:
    geometry = meta['geometry']
    settings = meta['settings']
    description = meta['description']
       
    # Translate:
    _translate_(geometry, records, geom_dict)
    _translate_(settings, records, sett_dict)
    _translate_(geometry, records, descr_dict)
        
    # Correct some records (FlexRay specific):
    geometry['det2obj'] = geometry.get('src2det') - geometry.get('src2obj')
    
    # binned pixel size can't be trusted in all logs... use voxel size:
    geometry['img_pixel'] *= _parse_unit_('um')
    geometry['det_pixel'] = numpy.round(geometry['img_pixel'] * (geometry['src2det'] / geometry['src2obj']), 5)    
    #geometry['det_pixel'] = geometry.get('det_pixel') * _parse_unit_('um')
    
    geometry['det_hrz'] += 24    
    geometry['src_vrt'] -= 5

    # Rotation axis:
    geometry['axs_hrz'] -= 0.5
    
    # Compute the center of the detector:
    roi = numpy.int32(geometry.get('roi').split(sep=','))
    geometry['roi'] = roi.tolist()

    centre = [(roi[0] + roi[2]) // 2 - 971, (roi[1] + roi[3]) // 2 - 767]
    
    # Take into account the ROI of the detector:
    geometry['det_vrt'] -= centre[1] / settings.get('binning') * geometry['det_pixel']
    geometry['det_hrz'] -= centre[0] / settings.get('binning') * geometry['det_pixel']
    
    geometry['vol_tra'][0] = (geometry['det_vrt'] * geometry['src2obj'] + geometry['src_vrt'] * geometry['det2obj']) / geometry.get('src2det')
    #reconstruction['img_pixel'] = geometry['det_pixel'] / (geometry['src2det'] / geometry['src2obj'])    
    
    # TODO: add support for piezo motors PI_X PI_Y
    
    # Populate meta:    
    meta = {'geometry':geometry, 'settings':settings, 'description':description}
        
    return meta

def _translate_(destination, source, dictionary):
    """
    Translate source records to destination.
    """
    for key in dictionary.keys():
        
        if dictionary[key] in source.keys():
            destination[key] = source[dictionary[key]]
            
        else:
            warnings.warn('Record is not found: ' + dictionary[key])

def _metadata_translate_(records):                  
    """
    Translate records parsed from the Flex-Ray log file (scan settings.txt) to the meta object
    """
    # If the file was not found:
    if records is None: return None
    
    # Initialize empty meta record:
    meta = create_meta(0, 0, 0, [0, 360], GEOM_STATIC_OFFSETS)
    
    # Dictionary that describes the Flexray log record:        
    geom_dict =     {'det_pixel':'detector pixel size',
                    
                    'src2obj':'sod',
                    'src2det':'sdd',
                    
                    'src_vrt':'ver_tube',
                    'src_hrz':'tra_tube',
                    
                    'det_vrt':'ver_det',
                    'det_hrz':'tra_det',                    
                    
                    'axs_hrz':'tra_obj',
                    
                    'theta_max':'last_angle',
                    'theta_min':'first_angle',
                    
                    'roi':'roi'}

    sett_dict =    {'voltage':'kv',
                    'power':'power',
                    'focus':'focusmode',
                    'averages':'averages',
                    'mode':'mode',
                    'filter':'filter',
                    
                    'exposure':'exposure',
                    
                    'dark_avrg' : 'dark',
                    'pre_flat':'pre_flat',
                    'post_flat':'post_flat'}
                    
    descr_dict =    {'duration':'total_scantime',
                    'name':'scan_name'}
        
    geometry = meta['geometry']
    settings = meta['settings']
    description = meta['description']
    
    # Translate:
    _translate_(geometry, records, geom_dict)
    _translate_(settings, records, sett_dict)
    _translate_(geometry, records, descr_dict)
        
    # Correct some records (FlexRay specific):
    geometry['det2obj'] = geometry.get('src2det') - geometry.get('src2obj')    
            
    geometry['det_hrz'] += 24    
    geometry['src_vrt'] -= 5

    # Rotation axis:
    geometry['axs_hrz'] -= 0.5
    
    # Compute the center of the detector:
    roi = re.sub('[] []', '', geometry['roi']).split(sep=',')
    roi = numpy.int32(roi)
    geometry['roi'] = roi.tolist()

    centre = [(roi[0] + roi[2]) // 2 - 971, (roi[1] + roi[3]) // 2 - 767]
    
    # Take into account the ROI of the detector:
    geometry['det_vrt'] -= centre[1] * geometry['det_pixel']
    geometry['det_hrz'] -= centre[0] * geometry['det_pixel']
    
    geometry['vol_tra'][0] = (geometry['det_vrt'] * geometry['src2obj'] + geometry['src_vrt'] * geometry['det2obj']) / geometry.get('src2det')
    geometry['img_pixel'] = geometry['det_pixel'] / (geometry['src2det'] / geometry['src2obj'])    
    
    # Populate meta:    
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
        
    # It looks like toml doesnt like numpy arrays. Use lists.
    # TODO: make nested:
    for key in meta.keys():
        if isinstance(meta[key], numpy.ndarray):
            #meta[key] = numpy.array2string(meta[key], suppress_small=True, separator=',')
            meta[key] = meta[key].tolist()
    
    # Save TOML to a file:
    with open(filename, 'w') as f:
        d = toml.dumps(meta, True)
        f.write(d)
        #toml.dump(meta, f)
        
def write_astra(filename, data_shape, meta):
    """
    Write an astra-readable projection geometry vector.
    """        
    geom = astra_proj_geom(meta, data_shape)
    
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
    try:
        meta = toml.load(file_path)
        
    except:
        warnings.warn('No meta file found at:'+file_path)
        meta = None    
    
    return meta
    
def shape_alike(vol1, vol2):
    '''
    Make sure two arrays have the same shape by padding:
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
    
    # Index of the left and right ramp:
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
            flexUtil.add_dim(array[right_sl], array[flexUtil.anyslice(array, -rampr-1, dim)])            
    
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
            flexUtil.add_dim(array[right_sl], array[flexUtil.anyslice(array, -rampr-1, dim)])            

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
 
def bin(array, dim = None):
    """
    Simple binning of the data:
    """ 
          
    if dim is not None:
        # apply binning in one dimension
        
        # First apply division by 2:
        if (array.dtype.kind == 'i') | (array.dtype.kind == 'u'):    
            array //= 2 # important for integers
        else:
            array /= 2
            
        if dim == 0:
             array[:-1:2, :, :] += array[1::2, :, :]
             return array[:-1:2, :, :]
             
        elif dim == 1:
             array[:, :-1:2, :] += array[:, 1::2, :]
             return array[:, :-1:2, :]
             
        elif dim == 2:
             array[:, :, :-1:2] += array[:, :, 1::2]
             return array[:, :, :-1:2]
             
    else:        
    
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

def medipix2astra(array):
    """
    Convert a given numpy array (sorted: index, hor, vert) to ASTRA-compatible projections stack
    """    
    # Don't apply ascontignuousarray on memmaps!    
    array = numpy.transpose(array, [2,0,1])
    
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
        
    # Use 'img_pixel' to override the voxel size:
    sample =  geometry.get('vol_sample')   
    voxel = numpy.array([sample[0], sample[1], sample[2]]) * geometry.get('img_pixel')

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
        
    #vol_geom = astra.creators.create_vol_geom(shape[1], shape[2], shape[0], 
    vol_geom = astra.create_vol_geom(shape[1], shape[2], shape[0], 
              -size[2]/2, size[2]/2, -size[1]/2, size[1]/2, 
              -size[0]/2 + offset, size[0]/2 + offset)
        
    return vol_geom   

def _modify_astra_vector_(proj_geom, geometry):
    """
    Modify ASTRA vector using known offsets from ideal circular geometry.
    """
    # Even if the geometry is of the type 'simple' (GEOM_SYMPLE), we need to generate ASTRA vector to be able to rotate the reconstruction volume if needed.
    # if geometry.get('type') == GEOM_SIMPLE:
    #    return proj_geom
    
    #proj_geom = astra.functions.geom_2vec(proj_geom)
    proj_geom = astra.geom_2vec(proj_geom)
    vectors = proj_geom['Vectors']
    
    theta_count = vectors.shape[0]
    det_pixel = geometry['det_pixel'] * numpy.array(geometry.get('proj_sample'))
    
    # Modify vector and apply it to astra projection geometry:
    for ii in range(0, theta_count):
        
        # Compute current offsets (for this angle):
        if geometry.get('type') == GEOM_SIMPLE:
            
            det_vrt = 0 
            det_hrz = 0
            det_mag = 0
            det_rot = 0
            src_vrt = 0
            src_hrz = 0
            src_mag = 0
            axs_hrz = 0
            axs_mag = 0
        
        # Compute current offsets:
        if geometry.get('type') == GEOM_STATIC_OFFSETS:
            
            det_vrt = geometry['det_vrt'] 
            det_hrz = geometry['det_hrz'] 
            det_mag = geometry['det_mag'] 
            det_rot = geometry['det_rot'] 
            src_vrt = geometry['src_vrt'] 
            src_hrz = geometry['src_hrz'] 
            src_mag = geometry['src_mag'] 
            axs_hrz = geometry['axs_hrz'] 
            axs_mag = geometry['axs_mag'] 
          
        # Use linear offsets:    
        elif geometry.get('type') == GEOM_LINEAR_OFFSETS:
            b = (ii / (theta_count - 1))
            a = 1 - b
            det_vrt = geometry['det_vrt'][0] * a + geometry['det_vrt'][1] * b
            det_hrz = geometry['det_hrz'][0] * a + geometry['det_hrz'][1] * b  
            det_mag = geometry['det_mag'][0] * a + geometry['det_mag'][1] * b  
            det_rot = geometry['det_rot'][0] * a + geometry['det_rot'][1] * b  
            src_vrt = geometry['src_vrt'][0] * a + geometry['src_vrt'][1] * b 
            src_hrz = geometry['src_hrz'][0] * a + geometry['src_hrz'][1] * b 
            src_mag = geometry['src_mag'][0] * a + geometry['src_mag'][1] * b 
            axs_hrz = geometry['axs_hrz'][0] * a + geometry['axs_hrz'][1] * b 
            axs_mag = geometry['axs_mag'][0] * a + geometry['axs_mag'][1] * b 
            
        else: raise ValueError('Wrong geometry type: ' + geometry.get('type'))

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
        det_vect += det_vrt * det_axis_vrt / det_pixel[0]

        #Detector shift (H):
        det_vect += det_hrz * det_axis_hrz / det_pixel[1]

        #Detector shift (M):
        det_vect += det_mag * det_normal /  det_pixel[1]

        #Source shift (V):
        src_vect += src_vrt * det_axis_vrt / det_pixel[0]

        #Source shift (H):
        src_vect += src_hrz * det_axis_hrz / det_pixel[1]

        #Source shift (M):
        src_vect += src_mag * det_normal / det_pixel[1] 

        # Rotation axis shift:
        det_vect -= axs_hrz * det_axis_hrz  / det_pixel[1]
        src_vect -= axs_hrz * det_axis_hrz  / det_pixel[1]
        det_vect -= axs_mag * det_normal /  det_pixel[1]
        src_vect -= axs_mag * det_normal /  det_pixel[1]

        # Rotation relative to the detector plane:
        # Compute rotation matrix
    
        T = transforms3d.axangles.axangle2mat(det_normal, det_rot)
        
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
        T = numpy.array([geometry['vol_tra'][1] * vect_norm / det_pixel[1], geometry['vol_tra'][2] * vect_norm / det_pixel[1], geometry['vol_tra'][0] * vect_norm / det_pixel[0]])    
        
        src_vect[:] -= numpy.dot(T, R)           
        det_vect[:] -= numpy.dot(T, R)
        
    proj_geom['Vectors'] = vectors
    
    return proj_geom

def astra_proj_geom(geometry, data_shape, index = None):
    """
    Generate the vector that describes positions of the source and detector.
    Works with three types of geometry: simple, static_offsets, linear_offsets.
    """   
    
    # Basic geometry:
    det_count_x = data_shape[2]
    det_count_z = data_shape[0]
    theta_count = data_shape[1]

    det_pixel = geometry['det_pixel'] * numpy.array(geometry.get('proj_sample'))
    
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
       
    #proj_geom = astra.creators.create_proj_geom('cone', det_pixel[1], det_pixel[0], det_count_z, det_count_x, thetas, src2obj, det2obj)
    proj_geom = astra.create_proj_geom('cone', det_pixel[1], det_pixel[0], det_count_z, det_count_x, thetas, src2obj, det2obj)
    
    # Modify proj_geom if geometry is of type: static_offsets or linear_offsets:
    proj_geom = _modify_astra_vector_(proj_geom, geometry)
    
    return proj_geom   
        
def detector_size(shape, geometry):
    '''
    Get the size of detector in mm.
    '''       
    return geometry['det_pixel'] * numpy.array(shape)

def volume_bounds(proj_shape, geometry):
    '''
    A very simplified version of volume bounds...
    '''
    # TODO: Compute this propoerly.... Dont trus the horizontal bounds!!!
    
    # Detector bounds:
    det_bounds = detector_bounds(proj_shape, geometry)
    
    # Demagnify detector bounds:
    fact = geometry['src2obj'] / (geometry['src2obj'] + geometry['det2obj'])
    vrt = numpy.array(det_bounds['vrt'])
    vrt_bounds = (vrt * fact + geometry['src_vrt'] * (1 - fact))
    
    hrz = numpy.array(det_bounds['hrz'])
    max_x = max(hrz - geometry['axs_hrz'])
    
    hrz_bounds = [geometry['vol_tra'][2] - max_x, geometry['vol_tra'][2] + max_x]
    mag_bounds = [geometry['vol_tra'][1] - max_x, geometry['vol_tra'][1] + max_x]
            
    vol_bounds = {'vrt':vrt_bounds, 'mag': mag_bounds, 'hrz': hrz_bounds}
    
    return vol_bounds

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
                 
def _read_tiff_(file, sample = 1, x_roi = [], y_roi = []):
    """
    Read a single image.
    """
        
    # Sometimes files dont have an extension. Fix it!
    # Here I will supress warnings, as some tif files give an annoying UseWarning but seem to work otherwise
    
    #warnings.filterwarnings("ignore")
    if os.path.splitext(file)[1] == '':
        #im = imageio.imread(file, format = 'tif', offset = 0)
        im = imageio.imread(file, format = 'tif')
    else:
        #im = imageio.imread(file, offset = 0)
        im = imageio.imread(file)
        
    #warnings.filterwarnings("default")        
        
    # TODO: Use kwags offset  and size to apply roi!
    if (y_roi != []):
        im = im[y_roi[0]:y_roi[1], :]
    if (x_roi != []):
        im = im[:, x_roi[0]:x_roi[1]]

    if sample != 1:
        im = im[::sample, ::sample]
    
    return im

def _sanity_check_(meta):
    
    minimum_set = ['det_pixel', 'src2det', 'src2obj', 'theta_max', 'theta_min', 'unit']

    for word in minimum_set:
        if (word not in meta['geometry']): raise ValueError('Missing ' + word + ' in the meta data. Something wrong with the log file?')
        
        #if type(meta['geometry'][word]) != float: raise ValueError('Wrong records in the meta data. Something wrong with the log file?', word, meta['geometry'][word])
   
def _file_to_dictionary_(path, file_mask, separator = ':'):
    '''
    Read text file and return a dictionary with records.
    '''
    
    # Initialize records:
    records = {}
    
    #names = []
    
    # Try to find the log file in the selected path and file_mask
    log_file = [x for x in os.listdir(path) if (os.path.isfile(os.path.join(path, x)) and file_mask in os.path.join(path, x))]

    # Check if there is one file:
    if len(log_file) == 0:
        warnings.warn('Log file not found in path: ' + path + ' *'+file_mask+'*')

        return None
        
    if len(log_file) > 1:
        print('Found several log files. Currently using: ' + log_file[0])
        log_file = os.path.join(path, log_file[0])
    else:
        log_file = os.path.join(path, log_file[0])

    # Loop to read the file record by record:
    with open(log_file, 'r') as logfile:
        for line in logfile:
            name, var = line.partition(separator)[::2]
            name = name.strip().lower()
            
            # Dont mind empty lines and []:
            if re.search('[a-zA-Z]', name):
                if (name[0] != '['):
                    
                    # Remove \n:
                    var = var.rstrip()
                    
                    # If needed to separate the var and save the number of save the whole string:               
                    try:
                        var = float(var.split()[0])
                        
                    except:
                        var = var
                        
                    records[name] = var
                    
    if not records:
        raise Exception('Something went wrong during parsing the log file at:' + path)                    
                
    return records                

def _parse_unit_(string):
    '''
    Look for units in the string and return a factor that converts this unit to Si.
    '''
    # Here is what we are looking for:
    units_dictionary = {'nm':1e-6, 'nanometre':1e-6, 'um':1e-3, 'micrometre':1e-3, 'mm':1, 
                        'millimetre':1, 'cm':10.0, 'centimetre':10.0, 'm':1e3, 'metre':1e3, 
                        'rad':1, 'deg':numpy.pi / 180.0, 'ms':1, 's':1e3, 'second':1e3, 
                        'minute':60e3, 'us':0.001, 'kev':1, 'mev':1e3, 'ev':0.001,
                        'kv':1, 'mv':1e3, 'v':0.001, 'ua':1, 'ma':1e3, 'a':1e6, 'line':1}    
                        
    factor = [units_dictionary[key] for key in units_dictionary.keys() if key in string.split()]
    
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
    
def delete_path(path):
    """
    Delete everything. Remove the evidence!
    """
    print('Deleting:', path)
    shutil.rmtree(path)
    
def _connect_sftp_(hostname, username, password, log_file):
    
    paramiko.util.log_to_file(log_file)

    # Open a transport
    transport = paramiko.Transport((hostname, 22))
    
    # Auth
    print('Authorizing sftp connection...')
    transport.connect(username = username, password = password)
    
    print('Done!')
    
    client = _MySFTPClient_.from_transport(transport)
    
    # Go!
    return client 
    
def ssh_get_path(hostname, username, password, local_path, remote_path):
    '''
    Get files and directories from that path...
    '''
    if not os.path.exists(local_path):
        os.mkdir(local_path)
        print('Local directory created:', local_path)
    
    # Connect to remote:
    sftp = _connect_sftp_(hostname, username, password, os.path.join(local_path, 'scp.log'))

    try:
        sftp.get_path(local_path, remote_path)            
        
    except:    
        sftp.close()
        raise Exception('SFTP connection error!')
        
    finally:        
        sftp.close()
    
def ssh_put_path(hostname, username, password, local_path, remote_path):
    '''
    Put files and directories to that path...
    '''
    if not os.path.exists(local_path):
        os.mkdir(local_path)
        #print('Local directory created:', local_path)
    
    # Connect to remote:
    sftp = _connect_sftp_(hostname, username, password, os.path.join(local_path, 'scp.log'))

    try:
        sftp.put_path(local_path, remote_path)            
        
    except:    
        sftp.close()
        raise Exception('SFTP connection error!')
        
    finally:
        sftp.close()
    
class _MySFTPClient_(paramiko.SFTPClient):
    '''
    Class needed for copying recursively through ssh (paramiko.SFTPClient only allowes to copy single files).
    '''
    _total_file_count_ = 0
    _current_file_count_ = 0
    
    def sftp_walk(self, remote):
        '''
        From https://gist.github.com/johnfink8/2190472
        '''
        # Kindof a stripped down  version of os.walk, implemented for 
        # sftp.  Tried running it flat without the yields, but it really
        # chokes on big directories.
        path=remote
        files=[]
        folders=[]
        for f in self.listdir_attr(remote):
            if stat.S_ISDIR(f.st_mode):
                folders.append(f.filename)
            else:
                files.append(f.filename)
        
        yield path,folders,files
        for folder in folders:
            new_path=os.path.join(remote,folder)
            for x in self.sftp_walk(new_path):
                yield x
    
    def _put_path_(self, local, remote):
        '''
        Recursive function for uploading directories.
        '''
        if not self._exists_remote_(remote):
            #print('*making:', remote)
            self.mkdir(remote, ignore_existing=True)
        
        for item in os.listdir(local):
            if os.path.isfile(os.path.join(local, item)):
                
                # Copy the file:
                self._current_file_count_ += 1
                
                # We will overwrite if file is bigger:
                #if not self._exists_remote_(os.path.join(remote, item)):
                if self._size_local_(os.path.join(local, item)) > self._size_remote_(os.path.join(remote, item)):
                    self.put(os.path.join(local, item), os.path.join(remote, item))
                
                flex.util.progress_bar(self._current_file_count_ / self._total_file_count_)
                
            else:
                
                #print('making:', os.path.join(remote, item))
                #self.mkdir(os.path.join(remote, item), ignore_existing=True)
                self._put_path_(os.path.join(local, item), os.path.join(remote, item))
    
    def put_path(self, local, remote):
        ''' Uploads the contents of the local directory to the remote path. The
            target directory needs to exists. All subdirectories in local are 
            created under remote.
        '''
        # Count all files:
        print('Counting files...')
        self._total_file_count_ = 0
        for root, subdirs, files in os.walk(local): self._total_file_count_ += len(files)
        
        print('Uploading %u files' % self._total_file_count_)
        
        # Upload files recursively:
        self._current_file_count_= 0
        self._put_path_(local, remote)

    def _get_path_(self, local, remote):
        """
        Recursive get method.
        """      
        
        # Create new dirs:
        if not os.path.exists(local):
            os.mkdir(local)
            #print('Local directory created:', local)
        
        # Copy files:
        for filename in self.listdir(remote):

            if stat.S_ISDIR(self.stat(os.path.join(remote, filename)).st_mode):
                
                # uses '/' path delimiter for remote server
                self._get_path_(os.path.join(local, filename), os.path.join(remote, filename))
                
            else:
                self._current_file_count_ += 1
                
                # Overwrite when bigger:
                if self._size_local_(os.path.join(local, filename)) < self._size_remote_(os.path.join(remote, filename)):
                #if not os.path.isfile(os.path.join(local, filename)):
                    
                    # Actual get has remote first:
                    self.get(os.path.join(remote, filename), os.path.join(local, filename))
                    
                    
                    flex.util.progress_bar((self._current_file_count_) / self._total_file_count_)
                    
    def get_path(self, local, remote):
        '''
        Download the content of the remote to the local path.
        '''
        if not self._exists_remote_(remote):
            print('Remote path doesnt exist :(((')
            return
        
        # Count all files:
        print('Counting files...')
        self._total_file_count_ = 0
        for root, subdirs, files in self.sftp_walk(remote): self._total_file_count_ += len(files)
        
        print('Downloading %u files...' % self._total_file_count_)
        
        # Upload files recursively:
        self._current_file_count_= 0
        self._get_path_(local, remote)              

    def _size_local_(self, path):
        try:
            sta = os.stat(path)
            
            return sta.st_size
        
        except IOError as e:
            if e.errno == errno.ENOENT:
                return 0
            raise
        
    def _size_remote_(self, path):
        try:
            sta = self.stat(path)
            
            return sta.st_size
        
        except IOError as e:
            if e.errno == errno.ENOENT:
                return 0
            raise
            
    def _exists_remote_(self, path):
        try:
            self.stat(path)
        except IOError as e:
            if e.errno == errno.ENOENT:
                return False
            raise
        else:
            return True
        
    
    def mkdir(self, path, mode=511, ignore_existing=False):
        ''' Augments mkdir by adding an option to not fail if the folder exists  '''
        try:
            super(_MySFTPClient_, self).mkdir(path, mode)
            
        except IOError:
            if ignore_existing:
                pass
            else:
                raise