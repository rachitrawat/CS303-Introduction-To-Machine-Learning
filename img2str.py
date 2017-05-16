# -*- coding: utf-8 -*-
"""
Created on Mon May  1 19:12:04 2017

@author: RudradeepGuha
"""

import cv2
import os
import dlib
import openface
from scipy.misc import imread

def get_face(img_file):

    predictor_model = "shape_predictor_68_face_landmarks.dat"
    
    face_detector = dlib.get_frontal_face_detector()
    face_pose_predictor = dlib.shape_predictor(predictor_model)
    face_aligner = openface.AlignDlib(predictor_model)
          
    image = imread("C:/Users/RudradeepGuha/Pictures/Camera Roll/" + img_file)
        
    if len(image.shape)==3:   
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
        #The 1 in the second argument indicates that we should upsample the image 1 time. This
        # will make everything bigger and allow us to detect more faces.
    detected_faces = face_detector(image, 1)    
    name = "win{0}".format(img_file)    
    name = dlib.image_window()
    name.set_image(image)
    
        # Loop through each face we found in the image
    for i, face_rect in enumerate(detected_faces):
          	 #Get the the face's pose
        pose_landmarks = face_pose_predictor(image, face_rect)
        name.add_overlay(face_rect)
        name.add_overlay(pose_landmarks)
    	    #Use openface to calculate and perform the face alignment
        alignedFace = face_aligner.align(534, image, face_rect, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        	 #Save the aligned image to a file
        cv2.imwrite(os.path.join("C:/Users/RudradeepGuha/Pictures/Camera Roll", "cropped_{}".format(img_file)), alignedFace)
            
        return "C:/Users/RudradeepGuha/Pictures/Camera Roll/cropped_{}".format(img_file)

def create_data(img_path):
    
    img = cv2.imread(img_path, 0)
    
    if len(img.shape)==3:   
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    dim = (48, 48)
     
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    cv2.imshow("resized", resized)
    cv2.waitKey(0)
    
    img_str = ' '.join(map(str, resized.flatten().tolist()))
    return img_str