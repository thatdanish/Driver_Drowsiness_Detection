import tensorflow as tf 
import keras
import cv2
#import numpy as np
import os
#import pygame

model_filepath = os.path.join('models/'+'model_cnn_2.h5')  #defining model path


model_cnn = keras.models.load_model(model_filepath) #loading model

face_det = cv2.CascadeClassifier('haar_cascades/frontal_face_alt.xml') #defining facial haar cascade variable
l_eye = cv2.CascadeClassifier('haar_cascades/lefteye_2splits.xml')  #defining left eye haar cascade 
r_eye = cv2.CascadeClassifier('haar_cascades/righteye_2splits.xml')  #definging right eye haar cascade

cap = cv2.VideoCapture(0)  #loading video from default source (webcam)
score = 0  #drowsiness score
pred_lefteye = [99]
pred_righteye = [99]
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
# Class Labels: Closed == 0 , Open == 1

while(True): #defining main detection loop (always TRUE)
    ret,frame = cap.read()
    height, width = frame.shape[:2]

    frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    face = face_det.detectMultiScale(frame_rgb,minNeighbors=2,scaleFactor=1.1,minSize=(25,25)) #detecting a full face from a single frame
    leye = l_eye.detectMultiScale(frame_rgb,minNeighbors=2) #detecting right eye from a single frame
    reye = r_eye.detectMultiScale(frame_rgb,minNeighbors=2) #detecting left eye from a single frame

  #  for (x,y,w,h) in face: #draw a rectangle around the frame when a face is detected
       
        #for (x,y,w,h) in face:
    cv2.rectangle(frame,(0,0),(width,height),(255,255,255),2)
    for (x,y,h,w) in leye: #extracting left from the frame 
        left_eye =frame[y:y+h,x:x+w]
        left_eye = cv2.resize(left_eye,(128,128))
        left_eye = left_eye.reshape(1,128,128,3)

        pred_lefteye = model_cnn.predict(left_eye)
        
        break

    for (x,y,w,h) in reye:  #extracting right eye from the frame
        right_eye = frame[y:y+h,x:x+w]
        right_eye = cv2.resize(right_eye,(128,128))
        right_eye = right_eye.reshape(1,128,128,3)

        pred_righteye = model_cnn.predict(right_eye)

        break
        
    if (pred_lefteye == 0 and pred_righteye == 0):  #checking the alertness
        score += 1
        cv2.putText(frame,'CLOSED',(10,height-20),font,1, (0,0,0),2)
    else:
        score -= 1
        cv2.putText(frame,'OPEN',(10,height-20),font,1,(0,0,0),2)
            
    if score < 0:  #fix: negative score error
        score = 0
    if score > 20: #action when driver is considered drowsy 
         cv2.rectangle(frame,(0,0),(width,height),(0,0,255),5)
         cv2.rectangle(frame,(0,0),(width,height),(255,255,255),5)

    cv2.putText(frame,"Score:"+str(score),(100,height-20),font,1,(0,0,0),2) 

    cv2.imshow('Drowsiness Detection',frame) #displaying

    if cv2.waitKey(1) == 27: #breaking out of the loop
        break


cap.release()
cap.destroyAllWindows()
