import tensorflow as tf 
import keras
import cv2
import time
import keyboard
#import numpy as np
import os
from pygame import mixer

model_filepath = os.path.join('model_dir/'+'model_cnn_2.h5')  #defining model path
sound_path = os.path.join('sounds/2sb.wav') #defining alarm sound file path
mixer.init()  

sound = mixer.Sound(sound_path)  #calling mixer instance 

model_cnn = keras.models.load_model(model_filepath) #loading model

face_det = cv2.CascadeClassifier('haar_cascades/frontal_face_alt.xml') #defining facial haar cascade variable
l_eye = cv2.CascadeClassifier('haar_cascades/lefteye_2splits.xml')  #defining left eye haar cascade 
r_eye = cv2.CascadeClassifier('haar_cascades/righteye_2splits.xml')  #definging right eye haar cascade

cap = cv2.VideoCapture(0)  #loading video from default source (webcam)
score = 100  #alertness score

pred_lefteye = [99]     #init value
pred_righteye = [99]  #init value

decision_bndr_scr = 75  #decision boundary for sounding the alarm
thick = 0
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

prev_frame_time = 0
new_frame_time = 0

# Class Labels: Closed == 0 , Open == 1

while(True): #defining main detection loop (always TRUE)
    ret,frame = cap.read()
    height, width = frame.shape[:2]

    frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    new_frame_time = time.time()
    

    face = face_det.detectMultiScale(frame_rgb,minNeighbors=2,scaleFactor=1.1,minSize=(25,25)) #detecting a full face from a single frame
    leye = l_eye.detectMultiScale(frame_rgb,minNeighbors=2) #detecting right eye from a single frame
    reye = r_eye.detectMultiScale(frame_rgb,minNeighbors=2) #detecting left eye from a single frame

    for (x,y,w,h) in face: #draw a rectangle around the frame when a face is detected
        cv2.rectangle(frame,(x,y),(x+w,y+h),(100,100,100),1)
       
       
    cv2.rectangle(frame,(0,height-35),(300,height-10),(0,0,0),thickness=cv2.FILLED) #for better displaying of text
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
        score -= 0.5
        cv2.putText(frame,'CLOSED,',(10,height-20),font,0.75, (255,255,255),1)
    else:
        score += 1
        cv2.putText(frame,'OPEN,',(10,height-20),font,0.75,(255,255,255),1)
            
    if score > decision_bndr_scr:  #fix: negative score error and stop alarm
        sound.stop()
        if score >100:
            score = 100
    if score < decision_bndr_scr: #action when driver is considered drowsy 
        cv2.putText(frame,'Press ENTER to turn off alarm',(100,50),font,1,(0,0,255),1)
        if score < 0:
            score = 0
               
        cv2.rectangle(frame,(0,0),(width,height),(0,0,255),5)
        sound.play() 
   

    cv2.putText(frame,"Alertness Level:"+str(score)+"%",(100,height-20),font,0.75,(255,255,255),1) 

    fps = 1/(new_frame_time-prev_frame_time)  #calculating FPS
    fps = str(int(fps))
    prev_frame_time = new_frame_time    
    cv2.putText(frame,fps,(width-20,15),font,0.65,(0,128,255),1,cv2.LINE_AA)  #displaying FPS in the rightside-up corner

    cv2.imshow('Drowsiness Detection',frame) #displaying
    
    if keyboard.is_pressed('ENTER'):    #score reset key (ENTER)
        score = 100
        print('***Score reset called.***')

    if cv2.waitKey(1) == 27:      #doomsday exit key (ESC)
        print('***Program terminated by the user.***')
        break
       

cap.release()
cv2.destroyAllWindows()
