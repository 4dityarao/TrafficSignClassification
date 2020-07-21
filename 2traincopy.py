import os
import cv2
#import pandas as pd
import pickle
import random
import numpy as np
#import matplotlib.pyplot as plt
import gc

PATH2= "C:\\Users\Aditya\\Desktop\\TEMProject\\Validation Set"
PATH = "C:\\Users\Aditya\\Desktop\\TEMProject\\Dataset"
training_data=[]
IMG_SIZE =32
#f = np.memmap('memmapped.dat', dtype=np.float64,mode='w+',shape=(10**2,81,81))
X_train=[]
y_train=[]
X_test=[]
y_test=[]



def create_training():

    for i in range(43):
        j=0

        str = PATH + "{}{}".format("\\", i)
        if not os.path.exists(str):
            print("Skipped",i)
            continue

        else:
            print(i)
            for file in os.listdir(str):
                #if(j>300):
                    #break


                #img_array = cv2.imread(os.path.join(str, file), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize( cv2.imread(os.path.join(str, file), cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE)) / 255.0
                #new_array=new_array/255
                #plt.imshow(new_array,cmap='gray')
                training_data.append([new_array, i])
                j+=1



create_training()

random.shuffle(training_data)


for features, label in training_data:
    X_train.append(features)
    y_train.append(label)
X_train=np.array(X_train).reshape(-1,IMG_SIZE,IMG_SIZE,1)
pickle_out = open("X_train.pickle", "wb")
pickle.dump(X_train, pickle_out)
pickle_out.close()
pickle_out = open("y_train.pickle", "wb")
pickle.dump(y_train, pickle_out)
pickle_out.close()
print("Done w training ")
print(y_train[:10])
gc.collect()


def create_validation() :

    for i in range(43):
        j=0
        str = PATH2 + "{}{}".format("\\", i)
        if not os.path.exists(str):
            print("Skipped",i)
            continue

        else:
            print(i)
            for file in os.listdir(str):
                if file == '{}.jpg'.format(i):
                    print("SKIPPED",file)
                    pass
                #if(j>90):
                    #break



                img_array = cv2.imread(os.path.join(str, file), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                new_array=new_array/255
                #plt.imshow(new_array,cmap='gray')
                training_data.append([new_array, i])
                j+=1





training_data=[]
create_validation()
random.shuffle(training_data)
for features, label in training_data:
    X_test.append(features)
    y_test.append(label)
X_test=np.array(X_test).reshape(-1,IMG_SIZE,IMG_SIZE,1)
print(y_test[:10])
pickle_out = open("X_test.pickle", "wb")
pickle.dump(X_test, pickle_out)
pickle_out.close()
pickle_out = open("y_test.pickle", "wb")
pickle.dump(y_test, pickle_out)
pickle_out.close()








