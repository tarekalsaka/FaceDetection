#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 17:28:36 2020

@author: francesco
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 16:38:28 2020

@author: tarek
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

anontation_path = "./FDDB/FDDB-folds/"
crop_face_list=[]
crop_backround_list=[]

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
      
# %%      
 
image_path, ann= readTextFile(anontation_path)
choose_image_to_work_on=image_path[10:15]
for index,x in enumerate(choose_image_to_work_on):
    print (index)
    faces, numFace,fullPath= parseAnnotation(choose_image_to_work_on[index],ann)
    drowElipse(fullPath,faces,numFace)
    # crop_faces, coord_crop_faces = crop_face(fullPath,faces,numFace)

