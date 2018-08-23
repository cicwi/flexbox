#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 2017
@author: kostenko

This module contains calculation routines for pre/post processing.
"""
import numpy
from scipy import ndimage
from scipy import signal
import transforms3d
import scipy.ndimage.interpolation as interp

from . import flexUtil
from . import flexData
from . import flexProject
from . import flexModel

def generate_stl(data, geometry):
    """
    Make a mesh from a volume.
    """
    
    from skimage import measure
    from stl import mesh

    # Segment the volume:
    threshold = binary_threshold(data, mode = 'otsu')
    
    # Close small holes:
    print('Filling small holes...')
    threshold = ndimage.morphology.binary_fill_holes(threshold, structure = numpy.ones((3,3,3)))

    print('Generating mesh...')
    # Use marching cubes to obtain the surface mesh of these ellipsoids
    verts, faces, normals, values = measure.marching_cubes_lewiner(threshold, 0)

    print('Mesh with %1.1e vertices generated.' % verts.shape[0])
    
    # Create stl:    
    stl_mesh = mesh.Mesh(numpy.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    stl_mesh.vectors = verts[faces] * geometry['img_pixel']
    
    return stl_mesh

def bounding_box(data):
    """
    Find a bounding box for the volume (use for auto_crop).
    """
    data = data.copy()
    
    soft_threshold(data, mode = 'otsu')

    integral = numpy.float32(data).sum(0)
    
    # Filter noise:
    integral = ndimage.gaussian_filter(integral, 10)
    mean = numpy.mean(integral[integral > 0])
    integral[integral < mean / 10] = 0
    
    # Compute bounding box:
    rows = numpy.any(integral, axis=1)
    cols = numpy.any(integral, axis=0)
    b = numpy.where(rows)[0][[0, -1]]
    c = numpy.where(cols)[0][[0, -1]]
    
    integral = numpy.float32(data).sum(1)
        
    # Filter noise:
    integral = ndimage.gaussian_filter(integral, 10)
    mean = numpy.mean(integral[integral > 0])
    integral[integral < mean / 10] = 0
    
    # Compute bounding box:
    rows = numpy.any(integral, axis=1)
    a = numpy.where(rows)[0][[0, -1]]
    
    return a, b, c

def soft_threshold(data, mode = 'histogram', threshold = 0):
    """
    Removes values smaller than the threshold value.
    
        mode (str)       : 'histogram', 'otsu' or 'constant'
        threshold (float): threshold value if mode = 'constant'
    """
    
    data[~binary_threshold(data, mode, threshold)] = 0
    
def binary_threshold(data, mode = 'histogram', threshold = 0):
    '''
    Compute binary mask that is True when values are above threshold. 
    Use 'histogram, 'otsu', or 'constant' mode.
    '''
    
    import matplotlib.pyplot as plt
    import skimage.filters
    
    print('Applying binary threshold...')
    
    if mode == 'otsu':
        threshold = skimage.filters.threshold_otsu(data[::2,::2,::2])    
        
    elif mode == 'histogram':
        x, y = histogram(data[::2,::2,::2], log = True, plot = False)
        
        # Make sure there are no 0s:
        y = numpy.log(y + 1)    
        y = ndimage.filters.gaussian_filter1d(y, sigma=1)
    
        # Find air maximum:
        air_index = numpy.argmax(y)
        
        print('Air found at %0.3f' % x[air_index])
    
        # Find the first shoulder after air peak in the histogram spectrum:
        x = x[air_index:]
        
        yd = abs(numpy.diff(y))
        yd = yd[air_index:]
        y = y[air_index:]
        
        # Minimum derivative = Saddle point or extremum:
        ind = signal.argrelextrema(yd, numpy.less)[0][0]
        min_ind = signal.argrelextrema(y, numpy.less)[0][0]
    
        plt.figure()
        plt.plot(x, y)
        plt.plot(x[ind], y[ind], '+')
        plt.plot(x[min_ind], y[min_ind], '*')
        plt.show()
        
        # Is it a Saddle point or extremum?
        if abs(ind - min_ind) < 2:    
            threshold = x[ind]         
    
            print('Minimum found next to the air peak at: %0.3f' % x[ind])        
        else:            
            # Move closer to the air peak since we are looking at some other material             
            threshold = x[ind] - abs(x[ind] - x[0]) / 4 
    
            print('Saddle point found next to the air peak at: %0.3f' % x[ind])        
            
    elif mode == 'constant':
        pass        
            
    else: raise ValueError('Wrong mode parameter. Can be histogram or otsu.')
    
    print('Threshold value is %0.3f' % threshold)
    
    return data > threshold
    
def _find_best_flip_(fixed, moving, Rfix, Tfix, Rmov, Tmov, use_CG = True, sample = 2):
    """
    Find the orientation of the moving volume with the mallest L2 distance from the fixed volume, 
    given that there is 180 degrees amiguity for each of three axes.
    
    Args:
        fixed(array): 3D volume
        moving(array): 3D volume
        centre(array): corrdinates of the center of rotation
        area(int): radius around the center of rotation to look at
        
    Returns:
        (array): rotation matrix corresponding to the best flip
    
    """
    fixed = fixed[::sample, ::sample, ::sample].copy()
    moving = moving[::sample, ::sample, ::sample].copy()
    
    # Apply filters to smooth erors somewhat:
    fixed = ndimage.filters.gaussian_filter(fixed, sigma = 2)
    moving = ndimage.filters.gaussian_filter(moving, sigma = 2)
    
    #flexUtil.display_projection(fixed - moving, title = 'before flips')
     
    # Generate flips:
    Rs = _generate_flips_(Rfix)
    
    # Compute L2 norms:
    Lmax = numpy.inf
    
    # Appliy flips:
    for ii in range(len(Rs)):
        
        Rtot_ = Rmov.T.dot(Rfix).dot(Rs[ii])
        Ttot_ = (Tfix - numpy.dot(Tmov, Rtot_)) / sample
        
        if use_CG:
            
            #flexUtil.display_projection(fixed - affine(moving, Rtot_, Ttot_), title = 'before itk')
            Ttot_, Rtot_, L = _itk_registration_(fixed, moving, Rtot_, Ttot_, shrink = [2,], smooth = [4,]) 
        
        L = norm(fixed - affine(moving, Rtot_, Ttot_))
        
        if Lmax > L:
            Rtot = Rtot_.copy()
            Ttot = Ttot_.copy()
            Lmax = L
            
            print('We found better flip(%u), L ='%ii, L)
            flexUtil.display_projection(fixed - affine(moving, Rtot_, Ttot_), title = 'Diff (%u). L2 = %f' %(ii, L))
    
    return Rtot, Ttot * sample 

def convolve_kernel(data, kernel):
    """
    Compute convolution with a kernel using FFT.
    """
    kernel = numpy.fft.fftshift(kernel)
    kernel = numpy.fft.fftn(kernel).conj()
    
    return numpy.real(numpy.fft.ifftn(numpy.fft.fftn(data) * kernel))

def find_marker(data, meta, r = 5):
    """
    Find a marker in 3D volume by applying a circular kernel with inner radius r [mm].
    """
    
    #data = data.copy()
    
    # Get areas with significant density:
    threshold = numpy.float32(binary_threshold(data, mode = 'otsu'))
    
    # Create a circular kernel:
    kernel = numpy.zeros_like(data)

    flexModel.phantom.sphere(kernel, meta['geometry'], r * 2, [0,0,0])
    kernel *= -0.5
    flexModel.phantom.sphere(kernel, meta['geometry'], r, [0,0,0])

    kernel[kernel > 0] *= ((r * 2)**3 - r**3) / r**3
    
    print('Computing feature sizes...')
    
    # Map showing the relative size of feature
    A = convolve_kernel(threshold, kernel)
    A /= A.max()
    
    print('Estimating local variance...')
    
    # Now estimate the local variance:
    B = ndimage.filters.laplace(data) ** 2    
    B /= data    
    
    # Make sure that boundaries don't affect variance estimation:
    threshold = threshold == 0
    
    threshold = ndimage.morphology.binary_dilation(threshold)
    
    B[threshold] = 0
    B = ndimage.filters.gaussian_filter(B, 3)
    B /= B.max()
    
    # Compute final weight:    
    A -= B
    
    index = numpy.argmax(A)
    
    # Display:
    flexUtil.display_max_projection(A, dim = 0, title = 'Marker map')
    
    # Coordinates:
    a, b, c = numpy.unravel_index(index, A.shape)
    
    print('Found the marker at:', a, b, c)
    
    return a, b, c
    
def moments_orientation(data, subsample = 1):
    '''
    Find the center of mass and the intensity axes of the image.
    
    Args:
        data(array): 3D input
        subsample: subsampling factor to to make it faster
        
    Returns:
        T, R: translation vector to the center of mass and rotation matrix to intensity axes 
    
    '''
    # find centroid:
    m000 = moment3(data, [0, 0, 0])
    m100 = moment3(data, [1, 0, 0])
    m010 = moment3(data, [0, 1, 0])
    m001 = moment3(data, [0, 0, 1])

    # Somehow this system of coordinates and the system of ndimage.interpolate require negation of j:
    T = [m100 / m000, m010 / m000, m001 / m000]
    
    # find central moments:
    mu200 = moment3(data, [2, 0, 0], T)
    mu020 = moment3(data, [0, 2, 0], T)
    mu002 = moment3(data, [0, 0, 2], T)
    mu110 = moment3(data, [1, 1, 0], T)
    mu101 = moment3(data, [1, 0, 1], T)
    mu011 = moment3(data, [0, 1, 1], T)
    
    # construct covariance matrix and compute rotation matrix:
    M = numpy.array([[mu200, mu110, mu101], [mu110, mu020, mu011], [mu101, mu011, mu002]])

    #Compute eigen vecors of the covariance matrix and sort by eigen values:
    vec = numpy.linalg.eig(M)[1].T
    lam = numpy.linalg.eig(M)[0]    
    
    # Here we sort the eigen values:
    ind = numpy.argsort(lam)
    
    # Matrix R is composed of basis vectors:
    R = numpy.array(vec[ind[::-1]])
    
    # Makes sure our basis always winds the same way:
    R[2, :] = numpy.cross(R[0, :], R[1, :])     
    
    # Centroid:
    T = numpy.array(T) - numpy.array(data.shape) // 2
    
    return T, R
        
def _itk2mat_(transform, shape):
    """
    Transform ITK to matrix and a translation vector.
    """
    
    # transform contains information about the centre of rptation, rotation and translation
    # We need to convert this to a rotation matrix and single translation vector
    # here we go,,,
    
    T = -numpy.array(transform.GetParameters()[3:][::-1])
    euler = -numpy.array(transform.GetParameters()[:3])
    R = transforms3d.euler.euler2mat(euler[0], euler[1], euler[2], axes='szyx')
    
    # Centre of rotation:
    centre = (transform.GetFixedParameters()[:3][::-1] - T)
    T0 = centre - numpy.array(shape) // 2
    
    # Add rotated vector pointing to the centre of rotation to total T
    T = T - numpy.dot(T0, R) + T0
    
    return T, R
    
def _mat2itk_(R, T, shape):
    """
    Initialize ITK transform from a rotation matrix and a translation vector
    """   
    import SimpleITK as sitk
    
    centre = numpy.array(shape, dtype = float) // 2
    euler = transforms3d.euler.mat2euler(R, axes = 'szyx')    

    transform = sitk.Euler3DTransform()
    transform.SetComputeZYX(True)
    
    transform.SetTranslation(-T[::-1])
    transform.SetCenter((centre + T)[::-1])    

    transform.SetRotation(-euler[0], -euler[1], -euler[2])    
    
    return transform    
   
def _moments_registration_(fixed, moving):
    """
    Register two volumes useing image moments.
    
        Args:
        fixed (array): fixed 3D array
        moving (array): moving 3D array
        
    Returns:
        moving will be altered in place.
        
        Ttot: translation vector
        Rtot: rotation matrix
        Tfix: position of the fixed volume

    """
    # Positions of the volumes:
    Tfix, Rfix  = moments_orientation(fixed)
    Tmov, Rmov  = moments_orientation(moving)
    
    flexUtil.progress_bar(0.1)
    
    # Total rotation and shift:
    Rtot = numpy.dot(Rmov, Rfix.T)
    Ttot = Tfix - numpy.dot(Tmov, Rtot)

    # Apply transformation:
    moving_ = affine(moving.copy(), Rtot, Ttot)
    
    # Solve ambiguity with directions of intensity axes:    
    Rtot, Ttot = _find_best_flip_(fixed, moving_, Rfix, Tfix, Rmov, Tmov, use_CG = False)
    
    return Ttot, Rtot, Tfix
    
def _itk_registration_(fixed, moving, R_init = None, T_init = None, shrink = [4, 1], smooth = [4, 0]):
    """
    Carry out ITK based volume registration (based on Congugate Gradient).
    
    Args:
        fixed (array): fixed 3D array
        moving (array): moving 3D array
        
    Returns:
        moving will be altered in place.
        
        T: translation vector
        R: rotation matrix
        
    """
    import SimpleITK as sitk
       
    #z = affine(moving, R_init, T_init)
    #flexUtil.display_projection(fixed, dim= 1, title = 'fixed') 
    #flexUtil.display_projection(z, dim= 1, title = 'before') 
            
    flexUtil.progress_bar(0) 
    
    # Initial transform:
    if R_init is None:
        R_init = numpy.zeros([3,3])
        R_init[0, 0] = 1
        R_init[1, 1] = 1
        R_init[2, 2] = 1
        
    if T_init is None:
        T_init = numpy.zeros(3)    
    
    # Initialize itk images:
    fixed_image =  sitk.GetImageFromArray(fixed)
    moving_image = sitk.GetImageFromArray(moving)
    
    # Regitration:
    registration_method = sitk.ImageRegistrationMethod()

    # Similarity metric settings.
    #registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)

    registration_method.SetInterpolator(sitk.sitkLinear)

    # Initial centering transform:
    transform = _mat2itk_(R_init, T_init, fixed.shape)   
    
    # Optimizer settings.
    registration_method.SetOptimizerAsPowell()
    #registration_method.SetOptimizerAsGradientDescent(learningRate=0.5, numberOfIterations=200, convergenceMinimumValue=1e-10, convergenceWindowSize=10)
    #registration_method.SetOptimizerAsGradientDescentLineSearch(learningRate=1, numberOfIterations = 100)
    #registration_method.SetOptimizerAsConjugateGradientLineSearch(learningRate=1, numberOfIterations = 100)
    #registration_method.SetOptimizerScalesFromPhysicalShift()

    # Setup for the multi-resolution framework.            
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors = shrink)
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas = smooth)
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Don't optimize in-place, we would possibly like to run this cell multiple times.
    registration_method.SetInitialTransform(transform, inPlace=False)

    transform = registration_method.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32), 
                                                  sitk.Cast(moving_image, sitk.sitkFloat32))
    
    flexUtil.progress_bar(1) 
    
    #print("Final metric value: ", registration_method.GetMetricValue())
    print("Optimizer`s stopping condition: ", registration_method.GetOptimizerStopConditionDescription())

    # This is a bit of woodo to get to the same definition of Euler angles and translation that I use:
    T, R = _itk2mat_(transform, moving.shape)
            
    #moving_image = sitk.Resample(moving_image, fixed_image, transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())
    #moving = sitk.GetArrayFromImage(moving_image)    
        
    #flexUtil.display_projection(fixed - moving, dim = 1, title = 'native diff')  
    
    return T, R, registration_method.GetMetricValue()
    
def affine(data, matrix, shift):
    """
    Apply 3x3 rotation matrix and shift to a 3D dataset.
    """
   
    # Compute offset:
    T0 = numpy.array(data.shape) // 2
    T1 = numpy.dot(matrix, T0 + shift)

    return ndimage.interpolation.affine_transform(data, matrix, offset = T0-T1, order = 1)
    
def _generate_flips_(Rfix):
    """
    Generate number of rotation and translation vectors.
    """    
    # Rotate the moving object around it's main axes:
    R = [numpy.eye(3),]
    
    # Axes:
    for ii in range(3):    
        #R.append(transforms3d.euler.axangle2mat(Rfix[ii], numpy.pi))
        # Angles:
        for jj in range(3):
            R.append(transforms3d.euler.axangle2mat(Rfix[ii], (jj+1) * numpy.pi/2))
    
    return R
                    
def register_volumes(fixed, moving, subsamp = 2, use_moments = True, use_CG = True, use_flips = False, threshold = 'otsu'):
    '''
    Registration of two 3D volumes.
    
    Args:
        fixed (array): reference volume
        moving (array): moving/slave volume
        subsamp (int): subsampling of the moments computation
        use_itk (bool): if True, use congugate descent method after aligning the moments
        treshold (str): can be None, 'otsu' or 'histogram' - defines the strategy for removing low intensity noise
        
    Returns:
        
    '''    
    import skimage.filters
    
    if fixed.shape != moving.shape: raise IndexError('Fixed and moving volumes have different dimensions:', fixed.shape, moving.shape)
    
    print('Using image moments to register volumes.')
        
    # Subsample volumes:
    fixed_0 = fixed[::subsamp,::subsamp,::subsamp].copy()
    moving_0 = moving[::subsamp,::subsamp,::subsamp].copy()
    
    # Histogram normalization:
    #fact_fix = numpy.mean(fixed_0[fixed_0 > 0.005])
    #fact_mov = numpy.mean(moving_0[moving_0 > 0.005])
    
    # Normalize:
    #fixed_0 /= fact_fix
    #moving_0 /= fact_mov
    
    if threshold:
        # We use Otsu here instead of binary_threshold to make sure that the same 
        # threshold is applied to both images:
        #fixed_0 = binary_threshold(fixed_0, threshold)
        #moving_0 = binary_threshold(moving_0, threshold)
        
        threshold = skimage.filters.threshold_otsu(numpy.appned(fixed_0[::2, ::2, ::2], moving_0[::2, ::2, ::2]))
        fixed_0 = fixed_0[fixed_0 < threshold]
        moving_0 = moving_0[moving_0 < threshold]
        
    L2 = norm(fixed_0 - moving_0)
    print('L2 norm before registration: %0.2e' % L2)
    
    if use_moments:
        
        print('Running moments registration.')
        flexUtil.progress_bar(0)
    
        # Positions of the volumes:
        Tfix, Rfix  = moments_orientation(fixed_0)
        Tmov, Rmov  = moments_orientation(moving_0)
               
        # Total rotation and shift:
        #Rtot = numpy.dot(Rmov, Rfix.T)
        #Rtot = Rmov.T.dot(Rfix)

        #Ttot = Tfix - numpy.dot(Tmov, Rtot)
        
        Rtot, Ttot = _find_best_flip_(fixed_0, moving_0, Rfix, Tfix, Rmov, Tmov, use_CG = use_flips)
        
        flexUtil.progress_bar(1)
    
    else:
        # Initial transform:
        Rtot = numpy.zeros([3,3])
        Rtot[0, 0] = 1
        Rtot[1, 1] = 1
        Rtot[2, 2] = 1
        
        Ttot = numpy.zeros(3)
            
    # Refine registration using ITK optimization:
    if use_CG:
        
        print('Running ITK optimization.')
        
        #Rtot = Rmov.T.dot(Rfix)
        #Rtot = Rmov.dot(Rfix.T)
        #Ttot = Tfix - Tmov.dot(Rtot)

        # Find flip with or without CG:
        #Rtot, Ttot = _find_best_flip_(fixed_0, moving_0, Rfix, Tfix, Rmov, Tmov, use_CG = use_flips)
        
        # Show the result of moments registration:
        L2 = norm(fixed_0 - affine(moving_0.copy(), Rtot, Ttot))
        print('L2 norm after moments registration: %0.2e' % L2)
            
        # Run CG with the best result:
        Ttot, Rtot, L = _itk_registration_(fixed_0, moving_0, Rtot, Ttot, shrink = [8, 2, 1], smooth = [8, 2, 0])               
            
    # Apply transformation:
    L2 = norm(fixed_0 - affine(moving_0.copy(), Rtot, Ttot))
    print('L2 norm after registration: %0.2e' % L2)
            
    print('Found shifts:', Ttot * subsamp)
    print('Found Euler rotations:', transforms3d.euler.mat2euler(Rtot))        
    
    return Rtot, Ttot * subsamp 
    
def transform_to_geometry(R, T, geom):
    """
    Transforms a rotationa matrix and translation vector. 
    """    
    # Translate to flex geometry:
    geom = geom.copy()
    geom['vol_rot'] = transforms3d.euler.mat2euler(R.T, axes = 'sxyz')
    geom['vol_tra'] = numpy.array(geom['vol_tra']) - numpy.dot(T, R.T)[[0,2,1]] * geom['img_pixel']
    
    return geom
    
def register_astra_geometry(proj_fix, proj_mov, geom_fix, geom_mov, subsamp = 1):
    """
    Compute a rigid transformation that makes sure that two reconstruction volumes are alligned.
    Args:
        proj_fix : projection data of the fixed volume
        proj_mov : projection data of the fixed volume
        geom_fix : projection data of the fixed volume
        geom_mov : projection data of the fixed volume
        
    Returns:
        geom : geometry for the second reconstruction volume
    """
    
    print('Computing a rigid tranformation between two datasets.')
    
    # Find maximum vol size:
    sz = numpy.array([proj_fix.shape, proj_mov.shape]).max(0)    
    vol1 = numpy.zeros(sz, dtype = 'float32')
    vol2 = numpy.zeros(sz, dtype = 'float32')
    
    flexProject.FDK(proj_fix, vol1, geom_fix)
    flexProject.FDK(proj_mov, vol2, geom_mov)
    
    #flexUtil.display_projection(vol1 - vol2, title = 'Before registration')
    
    # Find transformation between two volumes:
    R, T = register_volumes(vol1, vol2, subsamp = subsamp, use_moments=True, use_CG=True)
    
    #flexUtil.display_projection(vol1 - affine(vol2, R, T), title = 'Diff. After registration')
    #flexUtil.display_projection(vol1, title = 'V1. After registration')
    #flexUtil.display_projection(affine(vol2, R, T), title = 'V2. After registration')
    
    return R, T

def scale(data, factor, order = 1):
    '''
    Scales the volume via interpolation.
    '''
    print('Applying scaling.')
    
    flexUtil.progress_bar(0)  
    
    data = ndimage.interpolation.zoom(data, factor, order = order)
    
    flexUtil.progress_bar(1)      
    
    return data    
    
def rotate(data, angle, axis = 0):
    '''
    Rotates the volume via interpolation.
    '''
    
    print('Applying rotation.')
    
    flexUtil.progress_bar(0)  
    
    sz = data.shape[axis]
    
    for ii in range(sz):     
        
        sl = flexUtil.anyslice(data, ii, axis)
        
        data[sl] = ndimage.interpolation.rotate(data[sl], angle, reshape=False)
        
        flexUtil.progress_bar((ii+1) / sz)
        
    return data
        
def translate(data, shift, order = 1):
    """
    Apply a 3D tranlation.
    """
    
    print('Applying translation.')

    flexUtil.progress_bar(0)  
    
    ndimage.interpolation.shift(data, shift, output = data, order = order)
        
    flexUtil.progress_bar(1)   

    return data
    
def histogram(data, nbin = 256, rng = [], plot = True, log = False):
    """
    Compute histogram of the data.
    """
    
    #print('Calculating histogram...')
    
    if rng == []:
        mi = min(data.min(), 0)
        
        ma = numpy.percentile(data, 99.99)
    else:
        mi = rng[0]
        ma = rng[1]

    y, x = numpy.histogram(data, bins = nbin, range = [mi, ma])
    
    # Set bin values to the middle of the bin:
    x = (x[0:-1] + x[1:]) / 2

    if plot:
        flexUtil.plot(x, y, semilogy = log, title = 'Histogram')
    
    return x, y

def intensity_range(data):
    """
    Compute intensity range based on the histogram.
    
    Returns:
        a: position of the highest spike (typically air)
        b: 99.99th percentile
        c: center of mass of the histogram
    """
    # 256 bins should be sufficient for our dynamic range:
    x, y = histogram(data, nbin = 256, plot = False)
    
    # Smooth and find the first and the third maximum:
    y = ndimage.filters.gaussian_filter(numpy.log(y + 0.1), sigma = 1)
    
    # Air:
    a = x[numpy.argmax(y)]
    
    # Most of the other stuff:
    b = numpy.percentile(data, 99.99) 
    
    # Compute the center of mass excluding the high air spike +10% and outlayers:
    y = y[(x > a + (b-a)/10) & (x < b)]    
    x = x[(x > a + (b-a)/10) & (x < b)]
          
    c = numpy.sum(y * x) / numpy.sum(y)  
    
    return [a, b, c] 
    
def centre(data):
        """
        Compute the centre of the square of mass.
        """
        data2 = data.copy()**2
        
        M00 = data2.sum()
                
        return [moment2(data2, 1, 0) / M00, moment2(data2, 1, 1) / M00, moment2(data2, 1, 2) / M00]

def moment3(data, order, center = numpy.zeros(3), subsample = 1):
    '''
    Compute 3D image moments $mu_{ijk}$.
    
    Args:
        data(array): 3D dataset
        order(int): order of the moment
        center(array): coordinates of the center
        subsample: subsampling factor - 1,2,4...
        
    Returns:
        float: image moment
    
    '''
    # Create central indexes:
    shape = data.shape
       
    data_ = data[::subsample, ::subsample, ::subsample].copy()
    
    for dim in range(3):
        if order[dim] > 0:
            
            # Define moment:
            m = numpy.arange(0, shape[dim], dtype = numpy.float32)
            m -= center[dim]
                
            flexUtil.mult_dim(data_, m[::subsample] ** order[dim], dim)    
            
    return numpy.sum(data_) * (subsample**3)
    
def moment2(data, power, dim, centered = True):
    """
    Compute 2D image moments (weighed averages) of the data. 
    
    sum( (x - x0) ** power * data ) 
    
    Args:
        power (float): power of the image moment
        dim (uint): dimension along which to compute the moment
        centered (bool): if centered, center of coordinates is in the middle of array.
        
    """
    
    
    # Create central indexes:
    shape = data.shape

    # Index:        
    x = numpy.arange(0, shape[dim])    
    if centered:
        x -= shape[dim] // 2
    
    x **= power
    
    if dim == 0:
        return numpy.sum(x[:, None, None] * data)
    elif dim == 1:
        return numpy.sum(x[None, :, None] * data)
    else:
        return numpy.sum(x[None, None, :] * data)
    
def interpolate_lines(proj):
    '''
    Interpolate values of the horizontal read out lines of the flexray flat panel detector.
    '''
    
    lines = numpy.ones(proj.shape[0::2], dtype = bool)    
    
    sz = proj.shape[0]
    
    if sz == 1536:
        lines[125::256, :] = 0
        lines[126::256, :] = 0    
    else:
        step = sz // 12
        lines[(step-1)::step*2, :] = 0    

    interpolate_holes(proj, lines, kernel = [1,1])   
          
def interpolate_holes(data, mask2d, kernel = [1,1]):
    '''
    Fill in the holes, for instance, saturated pixels.
    
    Args:
        mask2d: holes are zeros. Mask is the same for all projections.
    '''
    mask_norm = ndimage.filters.gaussian_filter(numpy.float32(mask2d), sigma = kernel)
    #flexUtil.display_slice(mask_norm, title = 'mask_norm')
    
    sh = data.shape[1]
    flexUtil.progress_bar(0)        
    for ii in range(sh):    
            
        data[:, ii, :] = data[:, ii, :] * mask2d           

        # Compute the filler:
        tmp = ndimage.filters.gaussian_filter(data[:, ii, :], sigma = kernel) / mask_norm      
                                              
        #flexUtil.display_slice(tmp, title = 'tmp')

        # Apply filler:                 
        data[:, ii, :][~mask2d] = tmp[~mask2d]
        
        # Show progress:
        flexUtil.progress_bar((ii+1) / sh)
            
def expand_medipix(data):
    
    # Bigger array:
    sz = numpy.array(data.shape)
    sz[0] += 4
    sz[2] += 4
    new = numpy.zeros(sz, dtype = data.dtype)
    
    for ii in range(data.shape[1]):
        
        img = numpy.insert(data[: ,ii, :], 257, -1, axis = 0)
        img = numpy.insert(img, 256, -1, axis = 0)
        img = numpy.insert(img, 256, -1, axis = 0)
        img = numpy.insert(img, 255, -1, axis = 0)
    
        img = numpy.insert(img, 257-2, -1, axis = 1)
        img = numpy.insert(img, 256-2, -1, axis = 1)
        img = numpy.insert(img, 256-2, -1, axis = 1)
        img = numpy.insert(img, 255-2, -1, axis = 1)
        
        new[: ,ii, :] = img
        
    mask = img >= 0
    interpolate_holes(new, mask, kernel = [1,1])        
        
    return new            

def residual_rings(data, kernel=[3, 3]):
    '''
    Apply correction by computing outlayers .
    '''
    # Compute mean image of intensity variations that are < 5x5 pixels
    print('Our best agents are working on the case of the Residual Rings. This can take years if the kernel size is too big!')

    flexUtil.progress_bar(0)        
    
    tmp = numpy.zeros(data.shape[::2])
    
    for ii in range(data.shape[1]):                 
        
        block = data[:, ii, :]

        # Compute:
        tmp += (block - ndimage.filters.median_filter(block, size = kernel)).sum(1)
        
        flexUtil.progress_bar((ii+1) / data.shape[1])
        
    tmp /= data.shape[1]
    
    print('Subtract residual rings.')
    
    flexUtil.progress_bar(0)        
    
    for ii in range(data.shape[1]):                 
        
        block = data[:, ii, :]
        block -= tmp

        flexUtil.progress_bar((ii+1) / data.shape[1])
        
        data[:, ii, :] = block 
    
    print('Residual ring correcion applied.')

def subtract_air(data, air_val = None):
    '''
    Subtracts a coeffificient from each projection, that equals to the intensity of air.
    We are assuming that air will produce highest peak on the histogram.
    '''
    print('Air intensity will be derived from 10 pixel wide border.')
    
    # Compute air if needed:
    if air_val is None:  
        
        air_val = -numpy.inf
        
        for ii in range(data.shape[1]): 
            # Take pixels that belong to the 5 pixel-wide margin.
            
            block = data[:, ii, :]

            border = numpy.concatenate((block[:10, :].ravel(), block[-10:, :].ravel(), block[:, -10:].ravel(), block[:, :10].ravel()))
          
            y, x = numpy.histogram(border, 1024, range = [-0.1, 0.1])
            x = (x[0:-1] + x[1:]) / 2
    
            # Subtract maximum argument:    
            air_val = numpy.max([air_val, x[y.argmax()]])
    
    print('Subtracting %f' % air_val)  
    
    flexUtil.progress_bar(0)  
    
    for ii in range(data.shape[1]):  
        
        block = data[:, ii, :]

        block = block - air_val
        block[block < 0] = 0
        
        data[:, ii, :] = block

        flexUtil.progress_bar((ii+1) / data.shape[1])
                            
def _parabolic_min_(values, index, space):    
    '''
    Use parabolic interpolation to find the extremum close to the index value:
    '''
    if (index > 0) & (index < (values.size - 1)):
        # Compute parabolae:
        x = space[index-1:index+2]    
        y = values[index-1:index+2]

        denom = (x[0]-x[1]) * (x[0]-x[2]) * (x[1]-x[2])
        A = (x[2] * (y[1]-y[0]) + x[1] * (y[0]-y[2]) + x[0] * (y[2]-y[1])) / denom
        B = (x[2]*x[2] * (y[0]-y[1]) + x[1]*x[1] * (y[2]-y[0]) + x[0]*x[0] * (y[1]-y[2])) / denom
            
        x0 = -B / 2 / A  
        
    else:
        
        x0 = space[index]

    return x0    
    
def norm(array, type = 'L2'):
    """
    Compute L2 norm of the array.
    """
    return numpy.sqrt(numpy.mean((array)**2))    
    
def _modifier_l2cost_(projections, geometry, subsample, value, key = 'axs_hrz', display = False):
    '''
    Cost function based on L2 norm of the first derivative of the volume. Computation of the first derivative is done by FDK with pre-initialized reconstruction filter.
    '''
    geometry_ = geometry.copy()
    
    geometry_[key] = value

    vol = flexProject.sample_FDK(projections, geometry_, subsample)

    l2 = 0
    
    for ii in range(vol.shape[0]):
        grad = numpy.gradient(numpy.squeeze(vol[ii, :, :]))
        
        grad = (grad[0] ** 2 + grad[1] ** 2)         
        
        l2 += numpy.sum(grad)
        
    if display:
        flexUtil.display_slice(vol, title = 'Guess = %0.2e, L2 = %0.2e'% (value, l2))    
            
    return -l2    
    
def _optimize_modifier_subsample_(values, projections, geometry, samp = [1, 1], key = 'axs_hrz', display = True):  
    '''
    Optimize a modifier using a particular sampling of the projection data.
    '''  
    maxiter = values.size
    
    # Valuse of the objective function:
    func_values = numpy.zeros(maxiter)    
    
    print('Starting a full search from: %0.3f mm' % values.min(), 'to %0.3f mm'% values.max())
    
    ii = 0
    for val in values:
        func_values[ii] = _modifier_l2cost_(projections, geometry, samp, val, 'axs_hrz', display)

        ii += 1          
    
    min_index = func_values.argmin()    
    
    return _parabolic_min_(func_values, min_index, values)  
        
def optimize_rotation_center(projections, geometry, guess = None, subscale = 1, centre_of_mass = True):
    '''
    Find a center of rotation. If you can, use the center_of_mass option to get the initial guess.
    If that fails - use a subscale larger than the potential deviation from the center. Usually, 8 or 16 works fine!
    '''
    
    # Usually a good initial guess is the center of mass of the projection data:
    if  guess is None:  
        if centre_of_mass:
            
            print('Computing centre of mass...')
            guess = flexData.pixel2mm(centre(projections)[2], geometry)
        
        else:
        
            guess = geometry['axs_hrz']
        
    img_pix = geometry['det_pixel'] / ((geometry['src2obj'] + geometry['det2obj']) / geometry['src2obj'])
    
    print('The initial guess for the rotation axis shift is %0.3f mm' % guess)
    
    # Downscale the data:
    while subscale >= 1:
        
        # Check that subscale is 1 or divisible by 2:
        if (subscale != 1) & (subscale // 2 != subscale / 2): ValueError('Subscale factor should be a power of 2! Aborting...')
        
        print('Subscale factor %1d' % subscale)    

        # We will use constant subscale in the vertical direction but vary the horizontal subscale:
        samp =  [20, subscale, subscale]

        # Create a search space of 5 values around the initial guess:
        trial_values = numpy.linspace(guess - img_pix * subscale, guess + img_pix * subscale, 5)
        
        guess = _optimize_modifier_subsample_(trial_values, projections, geometry, samp, key = 'axs_hrz', display = False)
                
        print('Current guess is %0.3f mm' % guess)
        
        subscale = subscale // 2
    
    return guess

def process_flex(path, sample = 1, skip = 1, memmap = None):
    '''
    Read and process the data.
    
    Args:
        path:  path to the flexray data
        options: dictionary of options, such as bin (binning), memmap (use memmap to save RAM)
        
    Return:
        proj: min-log projections
        meta: meta data
        
    '''
    # Read:    
    print('Reading...')
    
    index = []
    proj, flat, dark, meta = flexData.read_flexray(path, skip = skip, sample = sample, memmap = memmap, index = index)
                
    # Show fow much memory we have:
    #flexUtil.print_memory()     
    
    # Prepro:
    print('Processing...')
    if dark.ndim > 2:
        dark = dark.mean(0)
        
    proj -= dark
    proj /= (flat.mean(0) - dark)
        
    numpy.log(proj, out = proj)
    proj *= -1
    
    # Fix nans and infs after log:
    proj[~numpy.isfinite(proj)] = 10
    
    proj = flexData.raw2astra(proj)    
    
    # Here we will also check whether all files were read and if not - modify thetas accordingly:
    index = numpy.array(index)
    index //= skip
    
    if (index[-1] + 1) != index.size:
        print(index.size)
        print(index[-1] + 1)
        print('Seemes like some files were corrupted or missing. We will try to correct thetas accordingly.')
        
        thetas = numpy.linspace(meta['geometry']['theta_min'], meta['geometry']['theta_max'], index[-1]+1)
        thetas = thetas[index]
        
        meta['geometry']['_thetas_'] = thetas
        
        import pylab
        pylab.plot(thetas, thetas ,'*')
        pylab.title('Thetas')
        
    # Show fow much memory we have:
    # flexUtil.print_memory()             
    
    return proj, meta

def medipix_quadrant_shift(data):
    '''
    Expand the middle line
    '''
    
    print('Applying medipix pixel shift.')
    
    # this one has to be applied to the whole dataset as it changes its size
    
    flexUtil.progress_bar(0)
    data[:,:, 0:data.shape[2]//2 - 2] = data[:,:, 2:data.shape[2]/2]
    data[:,:, data.shape[2]//2 + 2:] = data[:,:, data.shape[2]//2:-2]

    flexUtil.progress_bar(0.5)

    # Fill in two extra pixels:
    for ii in range(-2,2):
        closest_offset = -3 if (numpy.abs(-3-ii) < numpy.abs(2-ii)) else 2
        data[:,:, data.shape[2]//2 - ii] = data[:,:, data.shape[2]//2 + closest_offset]

    flexUtil.progress_bar(0.7)

    # Then in columns
    data[0:data.shape[0]//2 - 2,:,:] = data[2:data.shape[0]//2,:,:]
    data[data.shape[0]//2 + 2:, :, :] = data[data.shape[0]//2:-2,:,:]

    # Fill in two extra pixels:
    for jj in range(-2,2):
        closest_offset = -3 if (numpy.abs(-3-jj) < numpy.abs(2-jj)) else 2
        data[data.shape[0]//2 - jj,:,:] = data[data.shape[0]//2 + closest_offset,:,:]

    flexUtil.progress_bar(1)

    print('Medipix quadrant shift applied.')    
    
def _find_shift_(data_ref, data_slave, offset, dim = 1):    
    """
    Find a small 2D shift between two 3d images.
    """
    from skimage import feature
    import scipy.ndimage
     
    shifts = []
    
    # Look at a few slices along the dimension dim:
    for ii in numpy.arange(0, data_slave.shape[dim], 10):
        
        # Take a single slice:
        sl = flexUtil.anyslice(data_ref, ii, dim)    
        im_ref = numpy.squeeze(data_ref[sl]).copy()
        sl = flexUtil.anyslice(data_slave, ii, dim)    
        im_slv = numpy.squeeze(data_slave[sl]).copy()
        
        # Make sure that the data we compare is the same size:.        

        im_ref = im_ref[offset[0]:offset[0] + im_slv.shape[0], offset[1]:offset[1] + im_slv.shape[1]]
    
        # Find common area:        
        no_zero = (im_ref * im_slv) != 0

        if no_zero.sum() > 0:
            im_ref *= no_zero
            im_slv *= no_zero
            
            # Crop:
            im_ref = im_ref[numpy.ix_(no_zero.any(1),no_zero.any(0))]    
            im_slv = im_slv[numpy.ix_(no_zero.any(1),no_zero.any(0))]                

            #flexUtil.display_slice(im_ref - im_slv, title = 'im_ref')
                                  
            # Laplace is way better for clipped objects than comparing intensities!
            im_ref = scipy.ndimage.laplace(im_ref)
            im_slv = scipy.ndimage.laplace(im_slv)
        
            # Shift registration with subpixel accuracy (skimage):
            shift, error, diffphase = feature.register_translation(im_ref, im_slv, 10)
            
            if shift[0] is numpy.nan:
                flexUtil.display_slice(im_ref, title = 'ref')
                flexUtil.display_slice(im_slv, title = 'slv')
            
            shifts.append(shift)

    shifts = numpy.array(shifts)            
    
    if shifts.size == 0:        
        shift = [0, 0]
        
    else:
        # prune around mean:
        mean = numpy.mean(shifts, 0)    
        
        error = (shifts - mean[None, :])
        
        error = numpy.sqrt(error[:, 0] ** 2 + error[:, 1] ** 2)
        mean = numpy.sqrt(mean[None, 0]**2 + mean[None, 1]**2)
        
        shifts = shifts[error < mean]

        if shifts.size == 0:
            
            shift = [0, 0]
            
        else:
            
            # total:        
            shift = numpy.mean(shifts, 0)    
            std = numpy.std(shifts, 0)
            
            shift_norm = numpy.sqrt(shift[0]**2+shift[1]**2)
            std_norm = numpy.sqrt(std[0]**2+std[1]**2)
    
            print('Found shift:', shift, 'with STD:', std)
            
            # Check that std is at least 2 times less than the shift estimate:
            if (std_norm > shift_norm / 2)|(shift_norm < 1):    
                    print('Bad shift. Discarding it.')
                    shift = [0, 0]
                
    return shift 

def _append_(total, new, x_offset, y_offset, pad_x, pad_y, base_dist, new_dist):
    """
    Append a new image to total via interpolation:
    """
    
    # Pad to match sizes:
    new = numpy.pad(new.copy(), ((0, pad_y), (0, pad_x)), mode = 'constant')  
    
    # Apply shift:
    if (x_offset != 0) | (y_offset != 0):   
        
        # Shift image:
        new = interp.shift(new, [y_offset, x_offset], order = 1)
    
    # Create distances to edge:
    return ((base_dist * total) + (new_dist * new)) / norm
    
def append_tile(data, geom, tot_data, tot_geom):
    """
    Append a tile to a larger dataset.
    Args:
        
        data: projection stack
        geom: geometry descritption
        tot_data: output array
        tot_geom: output geometry
        
    """ 
        
    print('Stitching a tile...')               
    
    # Assuming all projections have equal number of angles and same pixel sizes
    total_shape = tot_data.shape[::2]
    det_shape = data.shape[::2]
    
    total_size = flexData.detector_size(total_shape, tot_geom)
    det_size = flexData.detector_size(det_shape, geom)
                    
    # Offset from the left top corner:
    x0 = tot_geom['det_hrz']
    y0 = tot_geom['det_vrt']
    
    x = geom['det_hrz']
    y = geom['det_vrt']
        
    x_offset = ((x - x0) + total_size[1] / 2 - det_size[1] / 2) / geom['det_pixel']
    y_offset = ((y - y0) + total_size[0] / 2 - det_size[0] / 2) / geom['det_pixel']
    
    # Round em up!            
    x_offset = int(numpy.round(x_offset))                   
    y_offset = int(numpy.round(y_offset))                   
                
    # Pad image to get the same size as the total_slice:        
    pad_x = tot_data.shape[2] - data.shape[2]
    pad_y = tot_data.shape[0] - data.shape[0]  
    
    # Collapce both datasets and compute residual shift
    shift = _find_shift_(tot_data, data, [y_offset, x_offset])
    
    x_offset += shift[1]
    y_offset += shift[0]
           
    flexUtil.progress_bar(0) 
    
    # Precompute weights:
    base0 = (tot_data[:, ::100, :].mean(1)) != 0
    
    new0 = numpy.zeros_like(base0)
    # Shift image:
    new0[:det_shape[0], :det_shape[1]] = 1.0
    new0 = interp.shift(new0, [y_offset, x_offset], order = 1)
    #new0[y_offset:int(y_offset+det_shape[0]), x_offset:int(x_offset + det_shape[1])] = 1.0
    
    base_dist = ndimage.distance_transform_bf(base0)    
    new_dist =  ndimage.distance_transform_bf(new0)    
     
    # Trim edges to avoid interpolation errors:
    base_dist -= 1    
    new_dist -= 1
    
    base_dist *= base_dist > 0
    new_dist *= new_dist > 0
    norm = (base_dist + new_dist)
    norm[norm == 0] = numpy.inf
    
    # Apply offsets:
    for ii in range(tot_data.shape[1]):   
        
        # Pad to match sizes:
        new = numpy.pad(data[:, ii, :], ((0, pad_y), (0, pad_x)), mode = 'constant')  
        
        # Apply shift:
        if (x_offset != 0) | (y_offset != 0):   
            
            # Shift image:
            new = interp.shift(new, [y_offset, x_offset], order = 1)
                    
        # Add two images in a smart way:
        base = tot_data[:, ii, :]  
        
        # Create distances to edge:
        tot_data[:, ii, :] = ((base_dist * base) + (new_dist * new)) / norm
        
        flexUtil.progress_bar((ii+1) / tot_data.shape[1])
        
def data_to_spectrum(path):
    """
    Convert data with Al calibration object at path to a spectrum.txt.
    """
    import os

    proj, meta = process_flex(path)
    
    flexUtil.display_slice(proj, dim=0,title = 'PROJECTIONS')

    vol = flexProject.init_volume(proj, meta['geometry'])
    
    flexProject.FDK(proj, vol, meta['geometry'])
    flexUtil.display_slice(vol, title = 'Uncorrected FDK')

    e, s = calibrate_spectrum(proj, vol, meta, compound = 'Al', density = 2.7, iterations = 1000, n_bin = 20)   

    file = os.path.join(path, 'spectrum.txt')
    numpy.savetxt(file, [e, s])
        
    
def calibrate_spectrum(projections, volume, meta, compound = 'Al', density = 2.7, threshold = None, iterations = 1000, n_bin = 10):
    '''
    Use the projection stack of a homogeneous object to estimate system's 
    effective spectrum.
    Can be used by process.equivalent_thickness to produce an equivalent 
    thickness projection stack.
    Please, use conventional geometry. 
    ''' 
    
    from . import flexSpectrum
    #import random
    
    geometry = meta['geometry']

    # Find the shape of the object:                                                    
    if threshold:
        segmentation = numpy.float32(binary_threshold(volume, mode = 'constant', threshold = threshold))
    else:
        segmentation = numpy.float32(binary_threshold(volume, mode = 'otsu'))
        
    # Crop:    
    #height = segmentation.shape[0]   
    #w = 15

    #length = length[height//2-w:height//2 + w, : ,:]    
    
    # Forward project the shape:                  
    print('Calculating the attenuation length.')  
    
    length = numpy.zeros_like(projections)    
    length = numpy.ascontiguousarray(length)
    flexProject.forwardproject(length, segmentation, geometry)
        
    intensity = numpy.exp(-projections)
    
    # Crop to avoid cone artefacts:
    height = intensity.shape[0]//2
    window = 10
    intensity = intensity[height-window:height+window,:,:]
    length = length[height-window:height+window,:,:]
    
    # Make 1D:
    intensity = intensity[length > 0].ravel()
    length = length[length > 0].ravel()
    
    lmax = length.max()
    lmin = length.min()    
    
    print('Maximum reprojected length:', lmax)
    print('Minimum reprojected length:', lmin)
    
    print('Selecting a random subset of points.')  
    
    # Rare the sample to avoid slow times:
    #index = random.sample(range(length.size), 1000000)
    #length = length[index]
    #intensity = intensity[index]
    
    print('Computing the intensity-length transfer function.')
    
    # Bin number for lengthes:
    bin_n = 128
    bins = numpy.linspace(lmin, lmax, bin_n)
    
    # Sample the midslice:
    #segmentation = segmentation[height//2-w:height//2 + w, : ,:]    
    #projections_ = projections[height//2-w:height//2 + w, : ,:]
    
    
    #import flexModel
    #ctf = flexModel.get_ctf(length.shape[::2], 'gaussian', [1, 1])
    #length = flexModel.apply_ctf(length, ctf)  
            
    # TODO: Some cropping might be needed to avoid artefacts at the edges
    
    #flexUtil.display_slice(length, title = 'length sinogram')
    #flexUtil.display_slice(projections_, title = 'apparent sinogram')
        
    # Rebin:
    idx  = numpy.digitize(length, bins)
    
    # Rebin length and intensity:        
    length_0 = bins + (bins[1] - bins[0]) / 2
    intensity_0 = [numpy.median(intensity[idx==k]) for k in range(bin_n)]
    
    # In case some bins are empty:
    intensity_0 = numpy.array(intensity_0)
    length_0 = numpy.array(length_0)
    length_0 = length_0[numpy.isfinite(intensity_0)]
    intensity_0 = intensity_0[numpy.isfinite(intensity_0)]

    # Get rid of tales:
    length_0 = length_0[5:-10]    
    intensity_0 = intensity_0[5:-10]    
    
    # Enforce zero-one values:
    length_0 = numpy.insert(length_0, 0, 0)
    intensity_0 = numpy.insert(intensity_0, 0, 1)
    
    #flexUtil.plot(length_0, intensity_0, title = 'Length v.s. Intensity')
        
    print('Intensity-length curve rebinned.')
        
    print('Computing the spectrum by Expectation Maximization.')
    
    energy = numpy.linspace(5, 100, n_bin)
    
    mu = flexSpectrum.linear_attenuation(energy, compound, density)
    exp_matrix = numpy.exp(-numpy.outer(length_0, mu))
    
    # Initial guess of the spectrum:
    spec = flexSpectrum.bremsstrahlung(energy, meta['settings']['voltage']) 
    spec *= flexSpectrum.scintillator_efficiency(energy, 'Si', rho = 5, thickness = 0.5)
    spec *= flexSpectrum.total_transmission(energy, 'H2O', 1, 1)
    spec *= energy
    spec /= spec.sum()
    
    #spec = numpy.ones_like(energy)
    #spec[0] = 0
    #spec[-1] = 0
    
    norm_sum = exp_matrix.sum(0)
    spec0 = spec.copy()
    #spec *= 0
    
    # SIRT type:
    import matplotlib.pyplot as plt
    '''
    w = exp_matrix.T.dot(exp_matrix.dot(spec + 1))
    w[w < 0.01] = numpy.inf
    
    print(w)
    
    for ii in range(iterations): 
        frw = exp_matrix.dot(spec)
        
        grad = exp_matrix.T.dot(intensity_0 - frw)
        spec = spec + grad / w
        spec[spec < 0] = 0
        
    '''    
    # EM type:
    
    for ii in range(iterations): 
        frw = exp_matrix.dot(spec)

        epsilon = frw.max() / 100
        frw[frw < epsilon] = epsilon

        spec = spec * exp_matrix.T.dot(intensity_0 / frw) / norm_sum

        # Make sure that the total count of spec is 1
        #spec = spec / spec.sum()
    
    
    print('Spectrum computed.')
        
    #flexUtil.plot(length_0, title = 'thickness')
    #flexUtil.plot(mu, title = 'mu')
    #flexUtil.plot(_intensity, title = 'synth_counts')
    
    # synthetic intensity for a check:
    _intensity = exp_matrix.dot(spec)
    
    # Display:       
    plt.figure()
    plt.semilogy(length[::200], intensity[::200], 'b.', lw=4, alpha=.8)
    plt.semilogy(length_0, intensity_0, 'g--')
    plt.semilogy(length_0, _intensity, 'r-', lw=3, alpha=.6)
    
    #plt.scatter(length[::100], -numpy.log(intensity[::100]), color='b', alpha=.2, s=4)
    plt.axis('tight')
    plt.title('Log intensity v.s. absorption length.')
    plt.legend(['raw','binned','solution'])
    plt.show() 
    
    # Display:
    plt.figure()
    plt.plot(energy, spec, 'b')
    plt.plot(energy, spec0, 'r:')
    plt.axis('tight')
    plt.title('Calculated spectrum')
    plt.legend(['computed','initial guess'])
    plt.show() 
            
    
    return energy, spec
    
def equivalent_density(projections, geometry, energy, spectrum, compound, density = 2, preview = True):
    '''
    Transfrom intensity values to projected density for a single material data
    '''
    # Assuming that we have log data!
    from . import flexSpectrum    
    import matplotlib.pyplot as plt

    print('Generating the transfer function.')
    
    # Attenuation of 1 mm:
    mu = flexSpectrum.linear_attenuation(energy, compound, density)
    
    # Make thickness range that is sufficient for interpolation:
    #m = (geometry['src2obj'] + geometry['det2obj']) / geometry['src2obj']
    #img_pix = geometry['det_pixel'] / m
    img_pix = geometry['img_pixel']

    thickness_min = 0
    thickness_max = max(projections.shape) * img_pix * 2
    
    print('Assuming thickness range:', [thickness_min, thickness_max])
    thickness = numpy.linspace(thickness_min, thickness_max, max(projections.shape))
    
    exp_matrix = numpy.exp(-numpy.outer(thickness, mu))
        
    synth_counts = exp_matrix.dot(spectrum)
    
    #flexUtil.plot(thickness, title = 'thickness')
    #flexUtil.plot(mu, title = 'mu')
    #flexUtil.plot(synth_counts, title = 'synth_counts')
    
    if preview:
        plt.figure()
        plt.semilogy(thickness, synth_counts, 'r-', lw=4, alpha=.8)
        plt.axis('tight')
        plt.title('Attenuation v.s. thickness [mm].')
        plt.show()
    
    synth_counts = -numpy.log(synth_counts)
    
    print('Callibration attenuation range:', [synth_counts[0], synth_counts[-1]])
    print('Data attenuation range:', [projections.min(), projections.max()])

    print('Applying transfer function.')    
    
    for ii in range(projections.shape[1]):
        
        projections[:, ii, :] = numpy.array(numpy.interp(projections[:, ii, :], synth_counts, thickness * density), dtype = 'float32') 
        flexUtil.progress_bar((ii+1) / projections.shape[1])        
        
    return projections