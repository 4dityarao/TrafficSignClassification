from flask import Flask, render_template, Response
import cv2
import pandas as pd
import numpy as np
import multiprocessing
import pickle
import playsound
import os
import time

arr=[0,1,7,10,12,29,26,17,19]
flag = 0
a = 43
prev_predict = 45
AUDIOPATH = "C:\\Users\\Aditya\\Desktop\\TEMProject\\Audio"
app = Flask(__name__)
def predictme(frame):

    frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    new_array = cv2.resize(np.asarray(frame2), (IMG_SIZE, IMG_SIZE)) / 255.0
    # plt.imshow(new_array,cmap='gray')
    # plt.show()
    new_array = new_array.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    prediction = model.predict_classes([new_array])
    p = model.predict([new_array])
    probability = np.amax(p)

    #Simply print the probability

    if probability>.9 and prediction[0] in arr:
        print(prediction, CATEGORIES[prediction[0]], probability)
        flag = 1
        return prediction[0]


cap = cv2.VideoCapture(0)
#PATH = "C:\\Users\Aditya\\Desktop\\TEMProject\\test"
IMG_SIZE=60
raw_data=pd.read_csv('signnamesLessClasses.csv')
CATEGORIES=raw_data['Name'].tolist()
pickle_in = open("Forty_FourV8(LessData).p", "rb")
model = pickle.load(pickle_in)
frame2 = cv2.imread("sample7.jpg")
i = cv2.selectROI(frame2)
cv2.destroyAllWindows()
roi = frame2[i[1]:i[1] + i[3], i[0]: i[0] + i[2]]
roi_hist = cv2.calcHist([roi], [0], None, [180], [0, 180])
term_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

@app.route('/')
def index():
    return render_template('index.html')

def gen():
    c = 0
    text = ""
    while True:


        ret, frame = cap.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180],
                                   1)  # Finds Largest Patch That Resembles HSV MASK OF ROI
        ret, track_window = cv2.CamShift(mask, (i[0], i[1], i[2], i[3]), term_criteria)
        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)
        cv2.polylines(frame, [pts], True, (255, 0, 0), 2)  # Draws Mask
        r = cv2.boundingRect(pts)  # Draws Rectangle Around the Mask
        # print(pts)
        roi2 = frame[r[1]:r[1] + r[3], r[0]:r[0] + r[2]]

        frame2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY)
        new_array = cv2.resize(np.asarray(frame2), (IMG_SIZE, IMG_SIZE)) / 255.0
        # plt.imshow(new_array,cmap='gray')
        # plt.show()
        new_array = new_array.reshape(1, IMG_SIZE, IMG_SIZE, 1)
        prediction = model.predict_classes([new_array])
        p = model.predict([new_array])
        probability = np.amax(p)


        if prediction[0] in arr:
            print(prediction, CATEGORIES[prediction[0]], probability)
            flag = 1
            a = prediction[0]
        else:
            a = 43


        if c == 100:
            print("VALUE GOING TO SOUND", a)
            text = CATEGORIES[a]
            playsound.playsound(os.path.join(AUDIOPATH, "{}.mp3").format(a), False)
            c = 0

        c += 1

        cv2.putText(frame, text, (26, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2,
                    cv2.LINE_AA, )



        if cv2.waitKey(1) and 0xFF == ord('q'):
            break


        formated_data = cv2.imencode('.jpeg', frame)[1]
        frame_bytes = formated_data.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='localhost',port='5000',threaded = False)