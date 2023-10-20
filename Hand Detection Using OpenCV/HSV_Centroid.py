# usr/bin/env python3
import cv2 as cv
import numpy as np

capture = cv.VideoCapture(0)

# Center crop for 720 * 1280 screen
height, width = 720, 1280
crop_height, crop_width = 600, 500
# wCam, hCam = 500,700 
# capture.set(3, wCam)
# capture.set(4, hCam)
# coordinates of the top-left corner of the crop
x1 = (width - crop_width) // 2
y1 = (height - crop_height) // 2

while True:

    isTrue, frame = capture.read()
    frame=cv.flip(frame,1)
    kernel = np.ones((3,3),np.uint8)

    cv.rectangle(frame, (x1, y1), (x1 + crop_width, y1 + crop_height), (0, 0, 255), 1)
    # roi = frame[160:560, 440:840]
    roi = frame[y1:y1+crop_height, x1:x1+crop_width]
    hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)

    lower_range = np.array([0, 20, 70], dtype=np.uint8)
    upper_range = np.array([20, 255, 255], dtype=np.uint8)

    mask = cv.inRange(hsv, lower_range, upper_range)

    #extrapolate the hand to fill dark spots within
    mask = cv.dilate(mask,kernel,iterations = 4)
        
    #blur the image
    mask = cv.GaussianBlur(mask,(5,5),100) 
    # cv.imshow("Masked",mask)
    contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Finding largest contour
    largest_contour = 0
    max_area = 1e-16
    for contour in contours:
        area = cv.contourArea(contour)
        if area > max_area:
            max_area = area
            largest_contour = contour

    if max_area > 100 :
        x, y, w, h = cv.boundingRect(largest_contour)
        cv.rectangle(roi, (x, y), (x + w, y + h), (255, 0, 255), 3)

        # Finding Centroid of hand
        Centroid = cv.moments(largest_contour)
        if (Centroid["m00"] != 0):
            cv.circle(roi, ( int(Centroid["m10"] / Centroid["m00"] + 1e-5), int(Centroid["m01"] / Centroid["m00"] + 1e-5) ), 5, (0, 0, 255), -1)
            # print(str(int(Centroid["m10"] / Centroid["m00"] + 1e-5)) +" " +str(int(Centroid["m01"] / Centroid["m00"] + 1e-5)))

                
    cv.imshow('Hand Detection', frame)

    if cv.waitKey(20) & 0xFF == ord('d'):
        break

capture.release()
cv.destroyAllWindows()
