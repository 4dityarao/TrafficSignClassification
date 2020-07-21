import cv2
import pandas as pd
#import tensorflow as tf
import numpy as np
from time import sleep
import pickle
import playsound
import os
import matplotlib.pyplot as plt

flag = 0
a = 43
prev_predict = 45
AUDIOPATH = "C:\\Users\\Aditya\\Desktop\\TEMProject\\Audio"
def  predictme(frame):

    frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    new_array = cv2.resize(np.asarray(frame2), (IMG_SIZE, IMG_SIZE)) / 255.0
    # plt.imshow(new_array,cmap='gray')
    # plt.show()
    new_array = new_array.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    prediction = model.predict_classes([new_array])
    p = model.predict([new_array])
    probability = np.amax(p)

    #Simply print the probability

    if probability>.9 and not prev_predict == prediction[0]:
        print(prediction, CATEGORIES[prediction[0]], probability)
        flag = 1
        return prediction[0]
    else:
        return 43



    '''
    probability = np.amax(p)
    if probability >.90:
        return prediction[0]
    else:
        return 42
    #sleep(3)
    '''


cap = cv2.VideoCapture(0)

#PATH = "C:\\Users\Aditya\\Desktop\\TEMProject\\test"
IMG_SIZE=60
raw_data=pd.read_csv('signnamesLessClasses.csv')
CATEGORIES=raw_data['Name'].tolist()
pickle_in = open("Forty_FourV8(LessData).p", "rb")
model = pickle.load(pickle_in)
frame2= cv2.imread("sample.jpg")
i=cv2.selectROI(frame2)
cv2.destroyAllWindows()
roi = frame2[i[1]:i[1] + i[3], i[0]: i[0]+i[2]]

# x=386
# y=203
# width=101
# height=111
# hsv_roi= cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
roi_hist=cv2.calcHist([roi], [0], None, [180], [0, 180])
term_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
a=True
while True:


    ret, frame=cap.read()
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    mask=cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1) # Finds Largest Patch That Resembles HSV MASK OF ROI
    ret, track_window = cv2.CamShift(mask, (i[0], i[1], i[2], i[3]), term_criteria)
    pts = cv2.boxPoints(ret)
    pts = np.int0(pts)
    cv2.polylines(frame, [pts], True, (255, 0, 0), 2) #Draws Mask
    r=cv2.boundingRect(pts)  # Draws Rectangle Around the Mask
    #print(pts)

    roi2= frame[r[1]:r[1]+r[3], r[0]:r[0]+r[2]]
    try:
        cv2.imshow("ROI", roi2)
    except cv2.error:
        print("ROI NOT FOUND")
        continue


    a= predictme(roi2)
    cv2.imshow("frame", cv2.putText(frame,CATEGORIES[a],(26,40),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2,cv2.LINE_AA,))
    #if flag ==1:
    playsound.playsound(os.path.join(AUDIOPATH,"{}.mp3").format(a))
    prev_predict = a
    flag = 0

    #cv2.imshow("Mask",mask)


    if cv2.waitKey(1) and 0xFF == ord('q'):
        break
    # #i=cv2.selectROI(frame)
    # #print(i)



cv2.waitKey(1)
cv2.destroyAllWindows()

#ROI(386, 203, 111, 101)
#(760, 405, 239, 222)

'''
while True:
    ret,frame = cap.read()
    #cv2.imshow('f', cv2.rectangle(frame,(152, 7), (152+338, 7+348),(255, 0, 0) ,2))
    r=(152, 7, 338, 348)
    #r= cv2.selectROI("Image", frame)
    x=np.array(frame)[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
    cv2.imshow('f2',x)
    c=0
    #for i in range(15):
        #a_prev = a
    predictme(x)
        #if(a_prev == a):
            #c+=1

    #if (c>5):
    #print(CATEGORIES[a])

    # frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # new_array = cv2.resize(np.asarray(frame2), (IMG_SIZE, IMG_SIZE)) / 255.0
    # # plt.imshow(new_array,cmap='gray')
    # # plt.show()
    # new_array = new_array.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    # prediction = model.predict_classes([new_array])
    # p = model.predict([new_array])
    # print(prediction)
    # print(CATEGORIES[prediction[0]])
    # probability = np.amax(p)
    # print(probability)
    predictme(frame)
    #print(CATEGORIES[a])





    if cv2.waitKey(1) and 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

'''