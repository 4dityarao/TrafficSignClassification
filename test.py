#import tensorflow as tf
import os
import cv2
import pandas as pd
import pickle
#import matplotlib.pyplot as plt
import playsound
import numpy as np
def testus():
    PATH = "C:\\Users\Aditya\\Desktop\\TEMProject\\test"
    IMG_SIZE=32
    raw_data=pd.read_csv('signnamesLessClasses.csv')
    CATEGORIES=raw_data['Name'].tolist()
    pickle_in = open("Forty_Three(08-5-20).p", "rb")

    model = pickle.load(pickle_in)
    '''for layer in model.layers:
        print(layer.get_weights())'''

    for file in os.listdir(PATH):
        img=cv2.imread(os.path.join(PATH,file),cv2.IMREAD_GRAYSCALE)
        new_array = cv2.resize(img, (IMG_SIZE, IMG_SIZE))/255.0
        #plt.imshow(new_array,cmap='gray')
        #plt.show()
        new_array = new_array.reshape(1, IMG_SIZE, IMG_SIZE, 1)
        prediction=model.predict_classes([new_array])
        p=model.predict([new_array])
        print(prediction)
        print(CATEGORIES[prediction[0]])
        print(np.argmax(p[0]))


testus()
