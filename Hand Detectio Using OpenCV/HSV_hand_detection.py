import cv2 as cv
import numpy as np

capture = cv.VideoCapture(1)

while True:
    ret, frame = capture.read()
    hsv_image = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    lower_limit = np.array([0,20,70])
    upper_limit = np.array([20,255,255])

    mask = cv.inRange(hsv_image, lower_limit, upper_limit)

    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # print(contours)
    for contour in contours:
        if cv.contourArea(contour) > 1000:  
            x, y, w, h = cv.boundingRect(contour)
            cv.rectangle(frame, (x, y), (x + w, y + h), (0,0,255), 2)

    cv.imshow("Hand Detection", frame)

    if cv.waitKey(20) & 0xFF==ord('d'):
        break
capture.release()
cv.destroyAllWindows()
