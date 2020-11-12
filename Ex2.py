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

def crop_face(image_path,faces,numberOfFaceInImage):
    '''
    this function take 
    
    image path: to read image
    faces : annotation of ellips 
    numberOfFaceInImage: number of face inside the image
                            to drow ellipce or exracted the face 
    return: the face image crop it from the image
            and left top right bottom of the image 
    '''
    img= cv2.imread(image_path)  
    crop_faces = []
    coord_crop_faces = []
    for i in range(numberOfFaceInImage):
        axis_radius1, axis_radius2, angle, center_x, center_y, _ = faces[i].split()
        axis_radius1= float(axis_radius1)
        axis_radius2= float(axis_radius2)
       
        
        major_axis = int(max(axis_radius1, axis_radius2))
        minor_axis = int(min(axis_radius1, axis_radius2))
        angle = int(math.degrees(float(angle))) 
        center_x, center_y = int(float(center_x)), int(float(center_y))
        
        temp_crop_face = img[check(center_y-major_axis):check(center_y+major_axis), check(center_x-minor_axis):check(center_x+minor_axis)]
        crop_faces.append(temp_crop_face)
        temp_coord_crop_face = check(center_x-minor_axis), check(center_y-major_axis), check(center_x+minor_axis), check(center_y+major_axis)
        coord_crop_faces.append(temp_coord_crop_face)
       
    return crop_faces, coord_crop_faces 

def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou
def check(value):
    if value < 0:
        value = 0
    return value

def check_division_by0(num, den):
    try:
        z = num / den
    except ZeroDivisionError:
        z = 0
    return z

# %%
from tensorflow.keras.models import load_model
# load model
model = load_model('my_model_trained.h5')
# summarize model.
model.summary()
# %%    
import time


Window_size = [16,19,23,28,33,40,48,57,69,83,99]
time_window = []
precision_evolution = []
recall_evolution = []
F1score_evolution =  []
for w in Window_size:
    start = time.time()
    image_path, ann= readTextFile(annotation_path)
    choose_image_to_work_on=image_path
    precision_window = []
    recall_window = []
    F1score_window = []
    for index,path_list in enumerate(choose_image_to_work_on): #iteration over the images
        if index % 50 == 0:
            print("image I work on ",index)
        faces, numFace,fullPath= parseAnnotation(path_list,ann)
        tmp_img = cv2.imread(fullPath)
        tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_RGB2GRAY) # put image in gray scale bc the network is trained in gray scale
#        plt.figure()
#        plt.imshow(tmp_img, cmap='gray') 
       
        #next reshaping is to add the channel value (1)
        tmp_img = tmp_img.reshape(tmp_img.shape[0], tmp_img.shape[1],1)
        
        
        crop_faces, coord_crop_faces = crop_face(fullPath,faces,numFace)
        
        dict_faces = {}   #initialization of a dictionary to understand which and how many faces will be recognised 
        for i in range(len(faces)):
            dict_faces[i] = 0
            
    #    print('inizializzazione del dizionario:', dict_faces)
        
        windowSize = w
        TP = 0 #true positive
        FP = 0 #false positive
        TN = 0 #true negative
        FN = 0 #false negative
        n_detected_faces = 0
        detected_faces = []
        crop_faces_corresponding = []
        false_negative = []
        missing_face = 0
        for (x, y, window) in sliding_window(tmp_img, stepSize=int(windowSize/4), windowSize=(windowSize,windowSize)):
            #reshaping the window for go in the network..
            window = cv2.resize(window, (24,16))
            window = window.reshape(1, window.shape[0], window.shape[1], 1)
            y_pred = model.predict(window)
            y_pred = np.argmax(y_pred, axis=1)
    #        n_faces = 0
            
            '''
                TRUE POSITIVE: 
                    window detect a face (y_pred == 0) and was a face (iou > 0.5)
                FALSE POSITIVE:
                    window detect a face (y_pred == 0) and was not a face (iou < 0.5)
                TRUE NEGATIVE: 
                    window detect a background (y_pred == 1) and was a background (iou < 0.5)
                FALSE NEGATIVE: 
                    window detect a background (y_pred == 1) and was a face (iou > 0.5)
            '''
            window_coord = x,y,x + windowSize, y + windowSize
    
            if (y_pred == 0): #the model predict that in the window there is a face
                for i in range(len(coord_crop_faces)):
                    iou = bb_intersection_over_union(coord_crop_faces[i], window_coord)
                    if iou > 0.5: #true positive
                        TP +=1
                        #distinguere le facce!!!!!!!!!
                        if dict_faces[i] == 0: #the face reconised from the model is memorized if before wasn't
#                            print('trovata una')
                            dict_faces[i] = 1
#                            n_detected_faces += 1
#                            detected_faces.append(window)
#                            crop_faces_corresponding.append(crop_faces[i])
                        #in case not if that face was already recognised
    
                    else:         #false positive
                        FP +=1
            else: #y_pred==1, the model predict that in the window there is a background
                for i in range(len(coord_crop_faces)):
                    iou = bb_intersection_over_union(coord_crop_faces[i], window_coord)
                    if iou > 0.5: #false negative
                        FN +=1
#                        false_negative.append(window)
#                        missing_face +=1
                    else:         #true negative
                        TN +=1
#        print('the number of the faces in the original image was ', numFace)
#        print('the model found ', n_detected_faces, ' faces')
#    
#        print('the faces detected were:')
#        for i in range(len(detected_faces)):
#            plt.figure()
#            plt.imshow(np.squeeze(detected_faces[i]), cmap='gray')
#            plt.figure()
#            plt.imshow(crop_faces_corresponding[i], cmap ='gray')
#           
#        print(TP)
#        print(FP)
#        print(TN)
#        print(FN)
        #  Report precision, recall, F1 and
        precision_tmp_img = check_division_by0(TP,(TP+FP))
        recall_tmp_img = check_division_by0(TP,(TP+FN))
        F1score_tmp_img = check_division_by0((2*(precision_tmp_img*recall_tmp_img)),(precision_tmp_img+recall_tmp_img))
        precision_window.append(precision_tmp_img)
        recall_window.append(recall_tmp_img)
        F1score_window.append(F1score_tmp_img)
        
    precision_evolution.append(np.mean(precision_window))  
    recall_evolution.append(np.mean(recall_window))
    F1score_evolution.append(np.mean(F1score_window))
time_window.append(time.time()-start)