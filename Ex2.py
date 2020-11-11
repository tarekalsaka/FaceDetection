# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 17:19:36 2020

@author: franc
"""



import cv2 
import numpy as np 
import matplotlib.pyplot as plt
import os 
import math


chdir=""
ann= []
all_image_path=[]
fullPath=[]
boxa=[]
annotation_path = "FDDB/FDDB-folds/"
iou=[]
numimgexclude=0 
allbackround=[]
allfaces=[]
imgExcludelist=[]

# %%
def readTextFile(path):
  list_of_files =sorted(os.listdir("./FDDB/FDDB-folds/"))

  for index, file in enumerate(list_of_files):
    #READ ALL THE ANNOTATION FILE AND STOR IT IN ONE LIST
    if index % 2 == 0:
      with open(annotation_path+list_of_files[index]) as ff:
          for xx in ff: 
              ann.append(xx.rstrip())
    else:
      with open(annotation_path+list_of_files[index]) as f:
        for xx in f:
          all_image_path.append(xx.rstrip())

  return all_image_path,ann


def parseAnnotation(image_path,annotationList):
  faces=[]
  
  image_Full_path= "./FDDB/originalPics/"+ image_path.strip()+".jpg"
  if image_path in annotationList:
    index=annotationList.index(image_path)
    numOfFace= int(annotationList[index+1])
    index=index+2
  for i in range(numOfFace):
     faces.append(annotationList[index].rstrip())
     index=index+1
  return  faces, numOfFace, image_Full_path

# %%
def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in range(0, image.shape[0]-windowSize[0], stepSize):
		for x in range(0, image.shape[1]-windowSize[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


# %%
from tensorflow.keras.models import load_model
# load model
model = load_model('my_model_trained.h5')
# summarize model.
model.summary()
# %%      
import imutils      
import time #to remove
Window_size = [16,19,23,28,33,40,48,57,69,83,99]

image_path, ann= readTextFile(annotation_path)
choose_image_to_work_on=image_path[1:3]
for index,path_list in enumerate(choose_image_to_work_on): #iteration over the images
    print("image I work on ",index)
    faces, numFace,fullPath= parseAnnotation(path_list,ann)
    tmp_img = cv2.imread(fullPath)
    tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_BGR2GRAY) # put image in gray scale bc the network is trained in gray scale
    
    plt.figure()
    plt.imshow(tmp_img)   
    tmp_img = tmp_img.reshape(tmp_img.shape[0], tmp_img.shape[1],1)
    print(tmp_img.shape)
    
    
    windowSize = 40
    for (x, y, window) in sliding_window(tmp_img, stepSize=int(windowSize/2), windowSize=(windowSize,windowSize)):
        #reshaping the window for go in the network..
        window = cv2.resize(window, (24,16))
        window = window.reshape(1, window.shape[0], window.shape[1], 1)
        y_pred = model.predict(window)
        y_pred = np.argmax(y_pred, axis=1)
        n_faces = 0
        if (y_pred == 1): #it's a face
            print('there is a face')
        
                
