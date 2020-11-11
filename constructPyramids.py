#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 15:40:25 2020

@author: tarek
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


imgFile = 'img_304.jpg'


def alg_pyramid(image, n_level, minSize=(10, 10)):
    pyramid = []
    sigma = 1
    #first level
    image = gaussian_blur(image, 13, sigma, verbose=False)

    pyramid.append(image)
    for i in range(n_level-1):
        sigma=1
        image = gaussian_blur(image, 13, sigma, verbose=False)
        sigma = sigma*np.sqrt(2)
        image = gaussian_blur(image, 13, sigma, verbose=False)
        image = downsampling(image)
        # if the resized image does not meet the supplied minimum
        # size, then stop constructing the pyramid
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break
        pyramid.append(image)

            
    return pyramid
