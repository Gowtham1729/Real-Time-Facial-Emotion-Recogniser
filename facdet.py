######################################################################

# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 09:42:38 2018

@author: Gowtham
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 22:26:34 2018

@author: gowtham
"""
import keras
#from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import  Dropout, Dense

#from keras.layers import Conv2D, MaxPooling2D
#from keras import backend as K
from keras.optimizers import Adadelta
import numpy as np
#import skimage.measure
#import pickle
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#from picl import make_keras_picklable

data = pd.read_csv('fer2013.csv')
X = data['pixels']
y = data['emotion']
y = np.array(y, dtype = np.uint8)
for i in range(len(y)):
    if y[i] == 1 or y[i] == 2:
        y[i] = 1
    elif y[i] == 3:
        y[i] = 2
    elif y[i] == 4:
        y[i] = 3
    elif y[i] == 5:
        y[i] = 4
    elif y[i] == 6:
        y[i] = 5
    

#y = y.reshape(len(y), 1)
data = None

Z = []
#K1 = 2
#L = 2
#MK = 48 // K1
#NL = 48 // L
for i in range(len(X)):
    a = np.fromstring(X[i],sep = ' ', dtype = int)#.reshape(48, 48)
    #a = a[:MK*K, :NL*L].reshape(MK, K1, NL, L).max(axis=(1, 3))
    #a = skimage.measure.block_reduce(X, (48, 48), np.max)
    #a.dtype = np.float32
    #a = a.reshape(48, 48)
    Z.append(a)
    
X = np.array(Z, dtype = np.uint8)
Z = None

y = np.array(y, dtype = np.uint8)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.05)

X = None
y = None

#batch_size = 128
num_classes = 6
#epochs = 12

# input image dimensions
#img_rows, img_cols = 48, 48

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



def ann(X_train, y_train, X_test, y_test):    
    
    
    """if K.image_data_format() == 'channels_first':
        X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)"""
    #make_keras_picklable()
   
    # Initialising the ANN
    classifier = Sequential()
    
    # Adding the input layer and the first hidden layer
    classifier.add(Dense(output_dim = 1024, init = 'uniform', activation = 'relu', input_dim = 2304))
    
    # Adding the second hidden layer
    classifier.add(Dense(output_dim = 1024, init = 'uniform', activation = 'relu'))
    classifier.add(Dropout(0.25))
    
    classifier.add(Dense(output_dim = 1024, init = 'uniform', activation = 'relu'))
    classifier.add(Dropout(0.25))
    
    
    classifier.add(Dense(output_dim = 1024, init = 'uniform', activation = 'relu'))
    
    
    classifier.add(Dense(output_dim = 1024, init = 'uniform', activation = 'relu'))
    
    
    classifier.add(Dense(output_dim = 1024, init = 'uniform', activation = 'relu'))
    classifier.add(Dropout(0.25))
    
    # Adding the output layer
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'sigmoid'))
    ada = Adadelta(lr=0.1, rho=0.95, epsilon=1e-08)
    # Compiling the ANN
    classifier.compile(optimizer = ada, loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    #summary of the neural network
    classifier.summary()
    
    # Fitting the ANN to the Training set
    classifier.fit(X_train, y_train, batch_size = 128, epochs = 25, validation_data=(X_test, y_test))
    
    #30 epochs seems to be ideal for getting 40% accuracy
    
    # add softmax function
    #classifier.add(Activation('softmax'))
    
    # optimizer is adadelta
    
    
    #compile the model
    
    
    
    # serialize model to JSON
    model_json = classifier.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    classifier.save_weights("model.h5")
    print("Saved model to disk")
    
    #pickle.dumps(classifier)
    
    #from keras.preprocessing.image import ImageDataGenerator
    
    """datagen = ImageDataGenerator(
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
    """
    # finally fit X_train and y_train
    
    score = classifier.evaluate(X_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    return classifier








########################################################################
classifier = ann(X_train, y_train, X_test, y_test)
########################################################################









#######################################################################
from keras.models import model_from_json
import numpy as np
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
#loaded_model = model_from_json(loaded_model_json)
#######################################################################
classifier = model_from_json(loaded_model_json)
import cv2

face_cascade = cv2.CascadeClassifier('/home/gowtham/anaconda2/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier('/home/gowtham/anaconda2/share/OpenCV/haarcascades/haarcascade_smile.xml')
#face_cascade = cv2.CascadeClassifier('C:\ProgramData\Anaconda3\pkgs\opencv-3.3.0-py36_200\Library\etc\haarcascades\haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

emotions = ['emojis/6.png','emojis/2.png','emojis/3.png','emojis/4.png','emojis/5.png','emojis/0_1.png',]
#emotion = {'Angry': 0, 'Disgust': 1, 'fear': 2, 'Happy': 3, 'Sad': 4, 'Surprise': 5, 'Neutral': 6}
emotion = {'Angry': 0, 'fear/digust': 1, 'Happy': 2, 'Sad': 3, 'Surprise': 4, 'Neutral': 5}
em = []
#em = np.array(em)
for i in range(len(emotions)):
    a = cv2.imread(emotions[i])
    em.append(a)

em = np.array(em)

while(True):
    ret,frame = cap.read()
    
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            #circ1 = cv2.circle(frame,(x+w/2,y+h/2),(min(w,h)/2)/2+50,(0,255,0),2)
            
            roi_gray = gray[y:y+h,x:x+w]
            roi_color = frame[y:y+h,x:x+w]
            
            smiles = smile_cascade.detectMultiScale(roi_gray,scaleFactor=1.7,minNeighbors=22,minSize=(25,25),flags=cv2.CASCADE_SCALE_IMAGE)
            for (a,b,c,d) in smiles:
                #print 2
                res_emo = cv2.resize(em[2],(w,h))
                frame[y:y+h, x:x+w] = res_emo
            
            #try:
            if len(smiles)==0:
                crop_img = gray[y:y+h, x:x+w]
            #This resized image is the input to your code
                resized_img = cv2.resize(crop_img,(48,48)).reshape(1,2304)
            #resized_img = np.array(resized_img)
            #Put your code here
            #Output of your code should be an integer between 0 and 6
            #print(classifier.predict(resized_img))
                i = np.argmax(classifier.predict(resized_img))
                #print i
                res_emo = cv2.resize(em[i],(w,h))
                frame[y:y+h, x:x+w] = res_emo
            
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()