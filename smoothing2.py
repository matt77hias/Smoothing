'''
Gaussian smoothing with Python and OpenCV.
@author     Matthias Moulin
@version    1.0
'''

import cv2
import math
import numpy as np
   
def gaussian_smooth2(img, sigma): 
    '''
    Does gaussian smoothing with sigma.
    Returns the smoothed image.
    @param     img          the image
    @param     sigma        the standard deviation
    @return    The smoothed image.
    '''
    result = np.zeros_like(img)
    
    #determine the length of the filter
    filter_length= math.ceil(sigma*5) 
    #make the length odd
    filter_length= 2*(int(filter_length)/2) +1  
            
    #Tip: smoothing=blurring, a filter=a kernel
    #or in a two step approach if you want the filter explicitly: getGaussianKernel & sepFilter2D
    return  cv2.GaussianBlur(img, (filter_length, filter_length), sigma)

#this part of the code is only executed if the file is run stand-alone
if __name__ == '__main__':
    #read an image
    img = cv2.imread('image.png')
    
    #show the image, and wait for a key to be pressed
    cv2.imshow('img', img)
    cv2.waitKey(0)
    
    #smooth the image
    smoothed_img = gaussian_smooth2(img, 2)
    
    #show the smoothed image, and wait for a key to be pressed
    cv2.imshow('smoothing2', smoothed_img)
    cv2.waitKey(0)
    cv2.imwrite('smoothing2.png', smoothed_img) 
    