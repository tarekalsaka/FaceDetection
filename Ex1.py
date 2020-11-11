# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 16:56:56 2020

@author: franc
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
from math import sqrt

imgFile = 'img_304.jpg'

# %%


def alg_pyramid(image, minSize=(16, 16)):
    pyramid = []
    sigma = 1
    #first level
    image = cv2.GaussianBlur(image, (7,7), sigma)

    pyramid.append(image)
    while True: #while  removing n_level
        sigma=1
        #kernal size (6*sigma)+1
        image = cv2.GaussianBlur(image, (7,7), sigma)
        sigma = sigma*np.sqrt(2)
    
        image = cv2.GaussianBlur(image, (9,9), sigma)
        w = int(image.shape[1] /2)
        h = int(image.shape[0] /2)
        image = imutils.resize(image, width=w, height=h)
        #image = cv2.resize(image, (image.shape[0]/2, image.shape[1]/2), interpolation= cv2.INTER_NEAREST)
        #image = image[::2,::2] #downsampling S_2
        # if the resized image does not meet the supplied minimum
        # size, then stop constructing the pyramid
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break
        pyramid.append(image)

            
    return pyramid



image = cv2.imread(imgFile)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.figure()
plt.imshow(image)
pyramid = alg_pyramid(image=image)

for i in range(len(pyramid)):
    print("Image Shape: {}".format(pyramid[i].shape))
    plt.figure()
    plt.imshow(pyramid[i])
        
#%%

from skimage.transform import pyramid_gaussian

delt = np.zeros((512, 512), dtype=np.float32)
u= []
# METHOD #2: Resizing + Gaussian smoothing.

py= pyramid_gaussian(delt, downscale=2)

for (i,inde)in enumerate(py):
    print (i)
    if inde.shape[0]<16 or inde.shape[1]<16:
        break
    u.append(inde)
# for (i, resized) in enumerate():
# 	# if the image is too small, break from the loop
# 	if resized.shape[0] < 30 or resized.shape[1] < 30:
# 		break
    

#%%
import skimage.exposure as exposure

# White image
delta = np.zeros((512, 512), dtype=np.float32)
# Dirac
delta[255,255] = 255

# sigmas
sigma1 = 1
sigma2 = sqrt(2)



# Pyramids
deltaPyramid = alg_pyramid(image = delta)

plt.figure()
plt.imshow(deltaPyramid[5])
deltaPyramidnor=[]
for index, img in enumerate(deltaPyramid):
    
    # norm_img = np.zeros((deltaPyramid[index].shape))
    # final_img = cv2.normalize(deltaPyramid[index],  norm_img, 0, 1, cv2.NORM_MINMAX)
    ii = cv2.GaussianBlur(img, (7,7), 1)
    

    deltaPyramidnor.append(ii)
    
    
for i in range(len(deltaPyramidnor)):
    centerx = deltaPyramid[i].shape[0]//2 #rows
    centery = deltaPyramid[i].shape[1]//2 #cols
    tmp_deltaPyramid = deltaPyramid[i][centerx, centery-6:centery+7]
    
    

# Impulse Response for each level
ImpResp = np.zeros((len(deltaPyramid), 13),dtype=float)
for idx, level in enumerate(deltaPyramid):
    centerx = level.shape[0]//2
    centery = level.shape[1]//2
    ImpResp[idx,:] = exposure.rescale_intensity(level[centerx, (centery-6):(centery+7)], out_range=(0,255), in_range='image').astype(np.uint8)

#visualize result
labels = []
for c in range(13):
    label = 'C{}'.format(c+1)
    labels.append(label)

x = np.arange(len(labels))  # the label locations
width = 0.1  # the width of the bars

fig, ax = plt.subplots()
rects1 = []
for k in range(ImpResp.shape[0]):
    rects1.append(ax.bar(x - 2*k*width, ImpResp[k], width, label='Level{}'.format(k)))

ax.set_ylabel('values')
ax.set_title('sigma0=1')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()


#%%
