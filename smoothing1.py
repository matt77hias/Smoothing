# -*- coding: utf-8 -*-
'''
Gaussian smoothing with Python.
@author     Matthias Moulin
@version    1.0
'''

import cv2
import numpy as np
import math

def gaussian_filter(sigma, filter_length=None):
    '''
    Given a sigma, returns a 1-D Gaussian filter.
    @param     sigma:         float, defining the width of the filter
    @param     filter_length: optional, the length of the filter, has to be odd
    @return    A 1-D numpy array of odd length,
               containing the symmetric, discrete approximation of a Gaussian with sigma
               Summation of the array-values must be equal to one.
    '''
    if filter_length==None:
        #determine the length of the filter
        filter_length = math.ceil(sigma*5)
        #make the length odd
        filter_length = 2 * (int(filter_length)/2) + 1

    #make sure sigma is a float
    sigma=float(sigma)

    #create the filter
    #uses Numpy array which gives better support for further calculations
    x_points =  np.array(range(1, filter_length // 2 + 1))
    #similar to the map function, uses Numpy's bordcasting rules
    f = np.vectorize(gaussian_value_function_with_fixed_sigma(sigma))

    result = f(x_points)
    result = np.concatenate((result[::-1], [f(0)], result))
    
    #When converting the Gaussian’s continuous values into the discrete values needed for a kernel,
    #the sum of the values will be different from 1. This will cause a darkening or brightening of
    #the image. To remedy this, the values can be normalized by dividing each term in the kernel
    #by the sum of all terms in the kernel.
    result = result / result.sum()
    #print(result)
    return result

def gaussian_value_function_with_fixed_sigma(sigma):
    '''
    This is just a wrapper for returning the gaussian_value function
    with a fixed standard deviation.
    @param     sigma       float, the standard deviation
    @return    A function that computes the Gaussian
               value for some x making use of sigma as standard
               deviation.
    '''

    def gaussian_value_with_fixed_sigma(x):
        '''
        Given a sigma, returns the Gaussian value for x.
        @param      x           the distance from the origin (in 0)
        @return     The Gaussian value for x.
        '''
        return gaussian_value(x, sigma)

    return gaussian_value_with_fixed_sigma

def gaussian_value(x, sigma):
    '''
    Given a sigma, returns the Gaussian value for x.
    @param      x           the distance from the origin (in 0)
    @param      sigma       float, the standard deviation
    @return     The Gaussian value for x.
    '''
    sigma2 = sigma*sigma
    return ((1 / math.sqrt(2*math.pi*sigma2)) * math.exp(-(x*x)/(2*sigma2)))

def test_gaussian_filter():
    '''
    Test the Gaussian filter on a known input.
    '''
    sigma = math.sqrt(1.0/2/math.log(2))
    f = gaussian_filter(sigma, filter_length=3)
    correct_f = np.array([0.25, 0.5, 0.25])
    error = np.abs( f - correct_f)
    if np.sum(error)<0.001:
        print "Congratulations, the filter works!"
    else:
        print "Still some work to do.."

def gaussian_smooth1(img, sigma):
    '''
    Does gaussian smoothing with sigma.
    Returns the smoothed image.
    @param     img          the image
    @param     sigma        the standard deviation
    @return    The smoothed image.
    '''
    result = np.zeros_like(img)

    #get the filter
    filtr = gaussian_filter(sigma)

    #smooth every color-channel
    for c in range(3):
        #smooth the 2D image img[:,:,c]
        result[:,:,c] = smooth(img[:,:,c], filtr)

    return result

def smooth(img, filtr):
    '''
    Smooths the given 2D image (only one color channel) with the given filter.
    Returns the smoothed 2D image.
    @param     img          the 2D image
    @param     filtr        the filter
    @return    The smoothed 2D image.
    '''
    #resolve the boundary effects by appending columns and rows
    nb_repeats = filtr.shape[0] // 2
    img = resolve_boundary_effects(img, nb_repeats, 0)
    img = resolve_boundary_effects(img, nb_repeats, 1)
    
    #mode 'same' returns output of length max(f_size, g_size). Boundary effects are still visible.
    #advantage of the Gaussian blur’s separable property by dividing the process into two passes.
    for i in range(img.shape[0]):
        img[i,:] = np.convolve(img[i,:], filtr, 'same')
    for j in range(img.shape[1]):
        img[:,j] = np.convolve(img[:,j], filtr, 'same')
    #restore the original size of the image
    #img' := [0, nb_repeats-1] union [nb_repeats, -nb_repeats-1] union [-nb_repeats, -1]
    #img := [nb_repeats, -nb_repeats)
    #[incl, excl)
    return img[nb_repeats:-nb_repeats, nb_repeats:-nb_repeats]
    
def resolve_boundary_effects(img, nb_repeats, axis=0):
    '''
    Adds nb_repeats boundaries on both sides to the given
    2D image (only one color channel) along the given axis.
    Returns the extended 2D image.
    @param     img          the 2D image
    @param     nb_repeats   the number of times to repeat on each side
    @param     axis         the axis along which to repeat
    @return    The smoothed 2D image.
    '''
    fix = np.ones(img.shape[axis], int)
    #repeat nb_repeats times the outermost columns/rows
    fix[0] = nb_repeats + 1
    fix[-1] = nb_repeats + 1
    return np.repeat(img, fix, axis)

#this part of the code is only executed if the file is run stand-alone
if __name__ == '__main__':
    #test the gaussian filter
    test_gaussian_filter()

    #read an image
    img = cv2.imread('image.png')

    #print the dimension of the image
    print img.shape

    #show the image, and wait for a key to be pressed
    cv2.imshow('img', img)
    cv2.waitKey(0)

    #smooth the image
    smoothed_img = gaussian_smooth1(img, 2)

    #show the smoothed image, and wait for a key to be pressed
    cv2.imshow('smoothing1', smoothed_img)
    cv2.waitKey(0)
    cv2.imwrite('smoothing1.png', smoothed_img) 
