# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 10:16:04 2018

@author: Gowtham
"""

#######################################################################
from keras.models import model_from_json
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
#loaded_model = model_from_json(loaded_model_json)
#######################################################################
classifier = model_from_json(loaded_model_json)
classifier.load_weights("model.h5")
import cv2

#face_cascade = cv2.CascadeClassifier('/home/gowtham/anaconda2/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier('C:\ProgramData\Anaconda3\pkgs\opencv-3.3.0-py36_200\Library\etc\haarcascades\haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

emotions = ['emojis/0_1.png','emojis/2.png','emojis/3.png','emojis/4.png','emojis/5.png','emojis/6.png',]
#emotion = {'Angry': 0, 'Disgust': 1, 'fear': 2, 'Happy': 3, 'Sad': 4, 'Surprise': 5, 'Neutral': 6}
emotion = {'Angry': 0, 'fear/digust': 1, 'Happy': 2, 'Sad': 3, 'Surprise': 4, 'Neutral': 5}

while(True):
    ret,frame = cap.read()
    
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x-25,y-25),(x+w+25,y+h+25),(255,0,0),2)
            #circ1 = cv2.circle(frame,(x+w/2,y+h/2),(min(w,h)/2)/2+50,(0,255,0),2)
            
            #try:
            crop_img = gray[y-35:y+h+25, x-25:x+w+25]
            #This resized image is the input to your code
            resized_img = cv2.resize(crop_img,(48,48)).reshape(1,2304)
            #resized_img = np.array(resized_img)
            #Put your code here
            #Output of your code should be an integer between 0 and 6
            #print(classifier.predict(resized_img))
            i = np.argmax(classifier.predict(resized_img))
            #print(i)
            emoji = cv2.imread(emotions[i])
            
            res_emo = cv2.resize(emoji,(w+50,h+50))
            frame[y-25:y+h+25, x-25:x+w+25] = res_emo
            #except:
                #None
            
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
