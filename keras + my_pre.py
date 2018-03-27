# -*- coding: utf-8 -*-
from __future__ import print_function

"""
Created on Sat Mar 10 12:49:17 2018

@author: Gowtham
"""

import keras
#from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.optimizers import Adadelta

import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

data = pd.read_csv('fer2013.csv')
X = data['pixels']
y = data['emotion']
y = y.reshape(len(y), 1)
data = None

Z = []

for i in range(len(X)):
    a = np.fromstring(X[i],sep = ' ', dtype = int)
    #a.dtype = np.float32
    a = a.reshape(48, 48)
    Z.append(a)
    
X = np.array(Z, dtype = np.uint8)
Z = None

y = np.array(y, dtype = np.uint8)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)

X = None
y = None

batch_size = 128
num_classes = 7
epochs = 40

# input image dimensions
img_rows, img_cols = 48, 48

# the data, split between train and test sets
#(X_train, y_train), (X_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255.0
X_test /= 255.0
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# add a sequential layer
model = Sequential()

# add first convolution layer
model.add(Conv2D(64, kernel_size = (5, 5) , border_mode='valid',
                        input_shape=input_shape))
#add first PRELu activation function
model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))

#take a padding
model.add(keras.layers.convolutional.ZeroPadding2D(padding=(2, 2), dim_ordering='th'))

#apply maxpool of size 5*5
model.add(MaxPooling2D(pool_size=(5, 5),strides=(2, 2)))

#take a padding
model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='th')) 

#add second convolution layer and repeat the above steps but chang the kernel size to 3*3
model.add(Conv2D(64, kernel_size = (3, 3)))
model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='th')) 

#add 3rd convolution layer
model.add(Conv2D(64, kernel_size = (3, 3)))
model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
model.add(keras.layers.convolutional.AveragePooling2D(pool_size=(3, 3),strides=(2, 2)))

model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='th'))

#add fourth convolution layer
model.add(Conv2D(128, kernel_size = (3, 3)))
model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='th'))

#add fifth convolution layer
model.add(Conv2D(128, kernel_size = (3, 3)))
model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))

model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='th'))
model.add(keras.layers.convolutional.AveragePooling2D(pool_size=(3, 3),strides=(2, 2)))

#flatten the output after average pooling and now size is reduced drastically
model.add(Flatten())

#1st neural network layer (input layer)
model.add(Dense(1024))
model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))#prelu as activation function
# add droput and remove some neurons for preventing overfitting
model.add(Dropout(0.2))

#add second neural network (1st layer)
model.add(Dense(1024))
model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))

# add another droput
model.add(Dropout(0.2))

# finally add output (with 7 dimensions such as 0,1,2,3,4,5,6)
model.add(Dense(7))

# add softmax function
model.add(Activation('softmax'))

# optimizer is adadelta
ada = Adadelta(lr=0.1, rho=0.95, epsilon=1e-08)

#compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=ada,
              metrics=['accuracy'])

#summary of the neural network
model.summary()
#(num, 48, 48, 1)
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=40,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images

datagen.fit(X_train)

import cv2
cv2.imshow('img', X_train[0].reshape(48, 48))
cv2.waitKey(0)

# finally fit X_train and y_train
model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test, y_test))

model_json = model.to_json()
with open("model1.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model1.h5")
print("Saved model to disk")


    
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#######################################################################
from keras.models import model_from_json
import numpy as np
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
#loaded_model = model_from_json(loaded_model_json)
#######################################################################
classifier = model_from_json(loaded_model_json) 
