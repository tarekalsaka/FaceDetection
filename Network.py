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
#from sklearn.model_selection import kfold


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
batch_size = 128
img_window = (16,24,1)
input_shape = np.prod(img_window)



def create_model(dropout_rate, neuron_1, neuron_2, neuron_3):
    optimizer = 'Adam'
    activation_function='relu'
    model = keras.Sequential(
        [            
            layers.InputLayer(input_shape=img_window),
            layers.Flatten(),
            layers.Dense(neuron_1, activation=activation_function),
            layers.Dropout(dropout_rate),
            layers.Dense(neuron_2, activation=activation_function),
            layers.Dropout(dropout_rate),
            layers.Dense(neuron_3, activation=activation_function),
            layers.Dense(2, activation="softmax")
        ]
    ) 
        
    
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]) #
    return model


#model.summary()

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
gridSearch = False
if gridSearch:
    from sklearn.model_selection import GridSearchCV
    from keras.wrappers.scikit_learn import KerasClassifier
    # create model
    model = KerasClassifier(build_fn=create_model, epochs=5, batch_size=100, verbose=0)
    
    #optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
    #activation = ['relu', 'tanh']
    
    dropout_rate = [0.1,0.2]
    neuron_1 = [256, 128]
    neuron_2 = [128, 64]
    neuron_3 = [32, 16]
    
    #optimizer, activation_function, dropout_rate, neuron_1, neuron_2, neuron_3
    
    param_grid = dict(dropout_rate=dropout_rate, neuron_1=neuron_1, neuron_2=neuron_2, neuron_3=neuron_3)
    grid = GridSearchCV(estimator=model, param_grid=param_grid,cv = 3)
    validation_generator_GS = valid_datagen.flow_from_directory(
            'val_data',
            target_size=img_window[:2],
            batch_size=828, # 828 so we can use grid.fit on all the validation dataset (1 batch)
            class_mode='categorical',
            color_mode='grayscale')
    X_val, Y_val = validation_generator_GS.next()
    
    grid_result = grid.fit(X_val, Y_val)
  
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

# %% TRAINING..
model_tuned = create_model(0.1, 256, 128, 16)
history = model_tuned.fit_generator(
        train_generator,
        steps_per_epoch = 500,
        epochs = 10,
        validation_data=validation_generator,
        validation_steps=200
        )
# %%
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# %%
test_datagen= ImageDataGenerator(rescale=1./255)
        
test_generator = test_datagen.flow_from_directory(
        'test_data',
        target_size=img_window[:2],
        batch_size=2070,
        class_mode='categorical',
        color_mode='grayscale',
        shuffle=False)
filenames = test_generator.filenames
nb_samples = len(filenames)
# %%
#ACCURACY
filenames = test_generator.filenames
nb_samples = len(filenames)
y_pred = model_tuned.predict_generator(generator=test_generator, steps = 1) # , steps=np.ceil(nb_samples/batch_size)
y_pred = np.argmax(y_pred, axis=1)
y_test = test_generator.classes
from sklearn.metrics import precision_score, accuracy_score, f1_score, recall
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
