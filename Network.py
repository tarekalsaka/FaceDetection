# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 15:07:44 2020

@author: franc
"""

import keras
from keras import layers
import numpy as np
import tensorflow as tf
from glob import glob
import matplotlib.pylab as plt
import os
import shutil

# %%
posCls = 'faces'
negCls = 'backgrounds'
root_dir = str(os.getcwd()) 
#os.makedirs(root_dir +'/train_data_' + posCls)
#os.makedirs(root_dir +'/train_data_' + negCls)
#os.makedirs(root_dir +'/val_data_' + posCls)
#os.makedirs(root_dir +'/val_data_' + negCls)
# %%
# Creating partitions of the data after shuffeling
#currentCls = posCls, negCls
#
#for i in range(len(currentCls)):
#    src = root_dir + '/' + currentCls[i] # Folder to copy images from
#    
#    
#    allFileNames = os.listdir(src)
#    np.random.shuffle(allFileNames)
#    train_FileNames, val_FileNames = np.split(np.array(allFileNames), [int(len(allFileNames)*0.9)])    
#    
#    train_FileNames = [src + "\\"  + name for name in train_FileNames.tolist()]
#    val_FileNames = [src+ '\\' + name for name in val_FileNames.tolist()]
#    
##    print('Total images: ', len(allFileNames))
##    print('Training: ', len(train_FileNames))
##    print('Validation: ', len(val_FileNames))
#    
#    # Copy-pasting images
#    for name in train_FileNames:
#        shutil.copy(name, root_dir + "/train_data_"+currentCls[i])
#    
#    for name in val_FileNames:
#        shutil.copy(name, root_dir + "/val_data_"+currentCls[i])
#    
#
##then create val_data folder and test_data folder and put the right folder inside

# %%

img_window = (8, 8, 1)
input_shape = np.prod(img_window)
batch_size = 128
epochs = 10


model = keras.Sequential(
    [
        layers.InputLayer(input_shape=img_window),
        layers.Flatten(),
        layers.Dense(32, activation="relu"),
        layers.Dense(16, activation="relu"),
        layers.Dense(8, activation="relu"),
        layers.Dense(2, activation="softmax"),
    ]        
)

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]) #
model.summary()

# %%

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range= 0.2,
        zoom_range=0.2,
        horizontal_flip=True)
valid_datagen= ImageDataGenerator(
        rescale=1./255,
        shear_range= 0.2,
        zoom_range=0.2,
        horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
        'train_data',
        target_size=img_window[:2],
        batch_size=batch_size,
        class_mode='categorical',
        color_mode='grayscale')
        
validation_generator = valid_datagen.flow_from_directory(
        'val_data',
        target_size=img_window[:2],
        batch_size=batch_size,
        class_mode='categorical',
        color_mode='grayscale')
        
# %%

model.fit_generator(
        train_generator,
        steps_per_epoch = 500,
        epochs = 2,
        validation_data=validation_generator,
        validation_steps=200
        )

# %%
class_names = glob("test_data\*")
class_names = [c.split("_") [-1] for c in class_names]

class_names = sorted(class_names)
name_id_map = dict(zip(range(len(class_names)), class_names))
name_id_map
# %%
test_datagen= ImageDataGenerator(rescale=1./255)
        
test_generator = test_datagen.flow_from_directory(
        'test_data',
        target_size=img_window[:2],
        batch_size=batch_size,
        class_mode='categorical',
        color_mode='grayscale')
        
model.evaluate_generator(generator=test_generator,steps=200)


