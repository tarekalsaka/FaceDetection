#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 17:28:36 2020

@author: tarek, francesco
"""

import cv2 
import numpy as np 
import matplotlib.pyplot as plt
import os 
import math
from random import randint


chdir="" 
ann= []
all_image_path=[]
fullPath=[]
directory_to_save_image='./faces'
boxa=[]
anontation_path = "./FDDB/FDDB-folds/"
iou=[]
numimgexclude=0 
allbackround=[]
allfaces=[]

def compare_tow_boxes(boxa,boxb, backround):
    
    b=0
    m=0
    print (len(boxb))
    while b < len(boxa):
        print ("b",b)
        while m < len(boxa) :
            iou1=bb_intersection_over_union(boxa[m],boxb[b])
            print ("M",m, iou1)
            m+=1
            iou.append(iou1)
        print (iou)
        if all(i<0.2 for i in iou):
            print ("hi")
            crop_backround_list.append(backround[b])
            b+=2
            iou.clear()
            m=0
        else :
            iou.clear()
            b+=2
            m=0
        if len(crop_backround_list)==numFace:
            print (len(crop_backround_list))
            break
    return crop_backround_list
def readTextFile(path):
  list_of_files =sorted(os.listdir("./FDDB/FDDB-folds/"))

  for index, file in enumerate(list_of_files):
    #READ ALL THE ANNOTATION FILE AND STOR IT IN ONE LIST
    if index % 2 == 0:
      with open(anontation_path+list_of_files[index]) as ff:
          for xx in ff: 
              ann.append(xx.rstrip())
    else:
      with open(anontation_path+list_of_files[index]) as f:
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
def check(value):
    if value < 0:
        value = 0
    return value
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
def drowElipse(image_path,faces,numberOfFaceInImage):
    
  img= cv2.imread(image_path)  
  for i in range(numberOfFaceInImage):
      axis_radius1, axis_radius2, angle, center_x, center_y, _ = faces[i].split()
      
      axis_radius1= float(axis_radius1)
      axis_radius2= float(axis_radius2)
      major_axis = int(max(axis_radius1, axis_radius2))
      minor_axis = int(min(axis_radius1, axis_radius2))
      angle = int(math.degrees(float(angle))) 
      center_x, center_y = int(float(center_x)), int(float(center_y))
      #draw the ellipse in the image
      cv2.ellipse(img, (center_x, center_y), (major_axis, minor_axis), angle, 0, 360, color=(0,0,255), thickness=3)
      
  plt.figure()
  plt.imshow(img)
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
def slide_box(shape,image,w,h,step):
    boxb.clear()
    back.clear()
    # for x in range(0,image.shape[1] - w , step):
    #         for y in range(0,image.shape[0] - h, step):
    #             window = image[x:x + w, y:y + h, :]
    #             wl, wt, wr, wb = x, y, x + w, y + h
    #             box=[wl, wt, wr, wb]
    #             if window.shape ==shape:
    #                 boxb.append(box)
    #                 back.append(window)
    x, y = np.random.randint(image.shape[0]-w, size=2)
    window= image[x:x+w, y:y+h] 
    wl, wt, wr, wb = x, y, x + w, y + h
    box=[wl, wt, wr, wb]
    boxb.append(box)
    back.append(window)
           

    return boxb,back                
def generate_random_box(crop_faces, fullPath, numFace):
    '''
    

    Parameters
    ----------
    crop_faces : list
        contain the faces image.
    fullPath : sting
        path to the image in the datasest to read.
    numFace : int
        number of face in each image.    

    Returns
    -------
    boxb : list 
        contain [wl, wt, wr, wb] for random box
        as much as #faces in image 
    temp_backround : list
         list of backround crop from image
         as much as #faces in image.
    ratio : float
        ratio of the image size and face size
        help to decide if the face as big as image
        then even ignore the image or reduce the size of box to half
        to be able to extract the backround not overlap this face.

    '''
    boxb=[]
    temp_backround=[]
    image = cv2.imread(fullPath)
    # plt.figure()
    # plt.imshow(image)

    for index in range(numFace):
        # plt.figure()
        # plt.imshow(crop_faces[index])
        ratio=  crop_faces[index].size/image.size
        w_face= crop_faces[index].shape[1]
        h_face=crop_faces[index].shape[0]
        x = np.random.randint(image.shape[1]-w_face)
        y=np.random.randint(image.shape[0]-h_face)
        window=image[y:y+h_face,x:x+w_face]
        wl, wt, wr, wb = x,y,x+w_face,y+h_face
        backbox_dim=[wl, wt, wr, wb]
        # facebox_dim=coord_crop_faces[index]
        
        boxb.append(backbox_dim)
        temp_backround.append(window)
        # print("face shape",crop_faces[index].shape)
        # print ("window shape",window.shape)
        
        # plt.figure()
        # plt.imshow(window)
     
    return boxb,temp_backround,ratio
def extract_backround(crop_faces, coord_crop_faces, fullPath, numFace):
    '''

    Parameters
    ----------
    crop_faces : list
        contain the faces image.
    coord_crop_faces : list 
        contain the (l,t,r,b) of the face.
    fullPath : string
        path to the image in the dataset to read.
    numFace : int
        number of face in each image.

    Returns
    -------
    list
        list with the backround image IOU with the face <0.2.

    '''
    crop_backround_list=[]
    # leastoverlap=[]
    while len(crop_backround_list)!= numFace:
        box,temp_back,retio= generate_random_box(crop_faces, fullPath, numFace)
        if retio>0.4:
            # with the same size of the face
            print ("impossibel to get backround")
            global numimgexclude
            numimgexclude+=1
            break
        for index, v in enumerate(box):
            for index1 , c in enumerate(coord_crop_faces):
                iou1=bb_intersection_over_union(c,v)
                # if iou1==0:
                    # leastoverlap.append((index,index1))
                iou.append(iou1)
        if all(i<0.2 for i in iou ):
            # print (leastoverlap)
            # print (iou)
            crop_backround_list.append(temp_back)
        else:
            # print ("backround match with face not accept")
            # leastoverlap.clear()
            iou.clear()
            # temp_back.clear()
            
    return crop_backround_list


#%%
image_path, ann= readTextFile(anontation_path)
choose_image_to_work_on=image_path[30:40]
for index,path_list in enumerate(choose_image_to_work_on):
    # print("image i work on ",index)
    faces, numFace,fullPath= parseAnnotation(path_list,ann)
    crop_faces, coord_crop_faces = crop_face(fullPath,faces,numFace)
    allfaces.append(crop_faces)
    # print ("number of face in image ",len(crop_faces))
    back= extract_backround(crop_faces,coord_crop_faces,fullPath,numFace)
    if  back:
        allbackround.append(back[0])
        
    
    
    
    
            
# for n , x in enumerate(crop_faces):
#     # filename = './faces/face{}{}.jpg'.format(index,n)
#     # cv2.imwrite(filename, x) 
#     plt.figure()
#     plt.imshow(x)
    
    
for n , x in enumerate(allfaces):
    for b,img in enumerate(allfaces[n]):
        # print(n,x)
        filename = './faces/back{}{}.jpg'.format(n,b)
        print(filename)
        plt.figure()
        plt.imshow(img)
        # cv2.imwrite(filename, img) 
        
for n , x in enumerate(allbackround):
    for b,img in enumerate(allbackround[n]):
        # print(n,x)
        filename = './backround/back{}{}.jpg'.format(n,b)
        print(filename)
        plt.figure()
        plt.imshow(img)
        # cv2.imwrite(filename, img) 
print("#of ex",numimgexclude)
        
        
            
            
            
            
            
        
                

        



    
    
    
    
    
    
    
    
    
    
    