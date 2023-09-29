import cv2 as cv
import numpy as np

capture = cv.VideoCapture(1)

while True:

    isTrue, frame = capture.read()

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

    if max_area > 100s :
        x, y, w, h = cv.boundingRect(largest_contour)
        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 3)

        # Finding Centroid of hand
        Centroid = cv.moments(largest_contour)
        if (Centroid["m00"] != 0):
            cv.circle(frame, ( int(Centroid["m10"] / Centroid["m00"] + 1e-5), int(Centroid["m01"] / Centroid["m00"] + 1e-5) ), 5, (0, 0, 255), -1)

    cv.imshow('Hand Detection', frame)

    if cv.waitKey(20) & 0xFF == ord('d'):
        break

capture.release()
cv.destroyAllWindows()