# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 13:34:00 2017

@author: RudradeepGuha
"""

from keras.models import load_model
import numpy as np
from img2str import create_data, get_face

emotion_code = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Sad', 5:'Surprise', 6:'Neutral'}
  
model = load_model("FER Train-72 Test-45.hdf5")

x = create_data(get_face("3.jpg"))
data = np.fromstring(x, dtype=int, sep=" ").reshape(1, 1, 48, 48)
prediction = model.predict(data)
p = prediction[0]
print(p)
m = max(p)
print(emotion_code.get(np.where(p==m)[0][0]))
correction = input("Was my prediction correct?[y/n] ")
if correction == 'y':
    print("Done")
else:
    print("Trying again....")
    new_arr = np.delete(p, np.where(p==m))
    nm = max(new_arr)
    print(emotion_code.get(np.where(p==nm)[0][0]))

    
