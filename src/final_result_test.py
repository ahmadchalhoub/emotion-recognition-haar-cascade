# Authors: Ahmad Chalhoub, Harjas Dadiyal, Geanna Perera

# Implementation of the trained Emotion Recognition model with
# Haar Cascade for face detection

# Note: the video frame capturing and result-outputting section of 
# this code was used from OpenCV's documentation. It wasn't fully written
# by us. The Haar Cascade section was also implemented from OpenCV's documentation

from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
from keras.models import model_from_json

face_classifier = cv2.CascadeClassifier('path to haarcascade_frontalface_default.xml file')
classifier = model_from_json(open("path to saved json file", "r").read())
classifier.load_weights('path to saved .h5 file')

class_labels=['Angry', 'Disgust', 'Fear', 'Happy','Sad','Surprised', 'Neutral']

# 0 - Angry
# 1 - Disgust
# 2 - Fear
# 3 - Happy
# 4 - Sad
# 5 - Surprised
# 6 - Neutral

# Implement Haar Cascade with the trained emotion recognition CNN

cap=cv2.VideoCapture(0)

while True:
    ret,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces=face_classifier.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray=gray[y:y+h,x:x+w]
        roi_gray=cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray])!=0:
            roi=roi_gray.astype('float')
            roi=img_to_array(roi)
            roi=np.expand_dims(roi,axis=0)
            preds=classifier.predict(roi)
            num_label = np.argmax(preds)
            label=class_labels[num_label]
            label_position=(x,y)
            cv2.putText(frame, label, label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        else:
            cv2.putText(frame,'No Face Found',(20,20),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
    
    cv2.imshow('Emotion Detector',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()