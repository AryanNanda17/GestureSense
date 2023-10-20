import cv2 as cv

capture = cv.VideoCapture(1)  

BackGroundSubtractor = cv.createBackgroundSubtractorMOG2()

while True:
    ret, frame = capture.read()

    ForeGroundMask = BackGroundSubtractor.apply(frame)
    # cv.imshow("foreground mask",ForeGroundMask)
    threshold, thresh = cv.threshold(ForeGroundMask, 200, 255, cv.THRESH_BINARY)
    # cv.imshow("threshold_image",thresh)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # print(contours)
    for contour in contours:
        if cv.contourArea(contour) > 1000:  
            x, y, w, h = cv.boundingRect(contour)
            cv.rectangle(frame, (x, y), (x + w, y + h), (0,0,255), 2)

    cv.imshow("Hand Detection", frame)

    if cv.waitKey(20) & 0xFF == ord('d'):
        break

captureture.release()
cv.destroyAllWindows()