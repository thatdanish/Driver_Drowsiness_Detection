import tensorflow as tf 
import keras
import cv2
import numpy as np
import os
import pygame

model_filepath = os.path.join('models/'+'model_cnn_2.h5')  #defining model path


model_cnn = keras.models.load_model(model_filepath) #loading model

face_det = cv2.CascadeClassifier('haar_cascades/frontal_face_alt.xml') #defining facial haar cascade variable
left_eye = cv2.CascadeClassifier('haar_cascade/lefteye_2splits.xml')  #defining left eye haar cascade 
right_eye = cv2.CascadeClassifier('haar_cascade/right_2splits.xml')  #definging right eye haar cascade

cap = cv2.VideoCapture(0)  #loading video from default source (webcam)

while(True):
    pass
