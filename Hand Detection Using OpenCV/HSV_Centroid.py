# usr/bin/env python3
import tensorflow as tf
from tensorflow import keras
import cv2 as cv
import numpy as np
import time
MODEL = tf.keras.models.load_model("/Users/Aryan/Documents/Projects/GestureSense/Models/1")

CLASS_NAMES = ['1finger', '2finger', '3finger', '4finger', 'kitli', 
              'neutral', 'pinch', 'pinky', 'snake', 'thumbsup', 'yoyo']

def predict(img):

    image = cv.resize(img,(128,128))
    image = image[:,:,:3]
    img_batch = np.expand_dims(image, 0)

    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    # return {
    #     'class': predicted_class,
    #     # 'confidence': float(confidence)
    # }
    return predicted_class

capture = cv.VideoCapture(1)
# Center crop for 720 * 1280 screen
# height, width = 720, 1280
# crop_height, crop_width = 640, 480
# coordinates of the top-left corner of the crop
# x1 = (width - crop_width) // 2
# y1 = (height - crop_height) // 2
# wCam, hCam = 640,480 
# capture.set(3, wCam)
# capture.set(4, hCam)
# FrameR = 100
while True:

    isTrue, frame = capture.read()
    # frame = cv.imread('/Users/Aryan/Documents/Projects/GestureSense/Hand_Recognition_Model/GestureDataset/DifferentGestures/pinch/gesture_8.jpg')
    print(frame.shape())
    frame=cv.flip(frame,1)
    # cv.rectangle(frame, (x1, y1), (x1 + crop_width, y1 + crop_height), (0, 0, 255), 1)
    # cv.rectangle(frame, (FrameR, FrameR), (wCam - FrameR, hCam - FrameR), (0, 0, 255), 1)
    # roi = frame[FrameR:hCam - FrameR, FrameR:wCam - FrameR]
    roi = frame
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    lower_range = np.array([0, 20, 70], dtype=np.uint8)
    upper_range = np.array([20, 255, 255], dtype=np.uint8)

    mask = cv.inRange(hsv, lower_range, upper_range)
    contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Finding largest contour
    largest_contour = 0
    max_area = 1e-16
    for contour in contours:
        area = cv.contourArea(contour)
        if area > max_area:
            max_area = area
            largest_contour = contour

    if max_area > 1000 :
        x, y, w, h = cv.boundingRect(largest_contour)
        # cv.rectangle(roi, (x, y), (x + w, y + h), (255, 0, 255), 3)
        hull = cv.convexHull(largest_contour)
        cv.drawContours(roi, [hull], 0, (255, 0, 255), 3)

        # Finding Centroid of hand
        Centroid = cv.moments(largest_contour)
        if (Centroid["m00"] != 0):
            cv.circle(roi, ( int(Centroid["m10"] / Centroid["m00"] + 1e-5), int(Centroid["m01"] / Centroid["m00"] + 1e-5) ), 5, (0, 0, 255), -1)
            # print(str(int(Centroid["m10"] / Centroid["m00"] + 1e-5)) +" " +str(int(Centroid["m01"] / Centroid["m00"] + 1e-5)))
    prediction = predict(roi)
    print(prediction)
    cv.putText(frame,str(prediction), (20,100),cv.FONT_HERSHEY_SIMPLEX,3, (0,0,255),2)
    cv.imshow("roi", mask)
    cv.imshow('Hand Detection', frame)
    # time.sleep(2)
    if cv.waitKey(20) & 0xFF == ord('d'):
        break

capture.release()
cv.destroyAllWindows()
