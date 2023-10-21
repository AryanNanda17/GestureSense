import cv2 as cv
import numpy as np

capture = cv.VideoCapture(1)
BackGroundSubtractor = cv.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)

while True:
    ret, frame = capture.read()
    ForeGroundMask = BackGroundSubtractor.apply(frame)

    # Set a learning rate to make the background adapt more gradually
    BackGroundSubtractor.apply(frame, learningRate=0.01)

    threshold, thresh = cv.threshold(ForeGroundMask, 200, 255, cv.THRESH_BINARY)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Finding largest contour
    largest_contour = None
    max_area = 1e-16
    for contour in contours:
        area = cv.contourArea(contour)
        if area > max_area:
            max_area = area
            largest_contour = contour

    if largest_contour is not None and max_area > 1000:
        # Create an empty mask image
        mask = np.zeros(frame.shape, dtype=np.uint8)

        # Draw the largest contour on the mask
        cv.drawContours(mask, [largest_contour], -1, (255, 255, 255), thickness=cv.FILLED)

        # Apply the mask to the original frame to get the masked image
        masked_image = cv.bitwise_and(frame, mask)

        x, y, w, h = cv.boundingRect(largest_contour)
        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 3)
        cv.imshow("Hand Detection", frame)
        cv.imshow("Masked Image", masked_image)

    if cv.waitKey(20) & 0xFF == ord('d'):
        break

capture.release()
cv.destroyAllWindows()
