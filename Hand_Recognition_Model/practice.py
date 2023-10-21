# import cv2 as cv
# import numpy as np 

# capture = cv.VideoCapture(1)
# while True:

#     frame = capture.read()
#     # frame = cv.imread('/Users/Aryan/Documents/Projects/GestureSense/Hand_Recognition_Model/GestureDataset/DifferentGestures/pinch/gesture_8.jpg')

    cv2.imshow("Video Frame", frame)
#     print(frame.shape)
#     if cv.waitKey(20) & 0xFF == ord('d'):
#         break

# capture.release()
# cv.destroyAllWindows()
import cv2

# Create a VideoCapture object to access the camera (0 for the default camera)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    if not ret:
        print("Error reading the video feed")
        break

    # Get the shape of the frame (height, width, and number of channels)
    height, width, channels = frame.shape

    # Display the video frame
    cv2.imshow("Video Frame", frame)

    # Print the shape of the frame
    # print(f"Frame Shape: Height={height}, Width={width}, Channels={channels}")
    print(frame.shape)

    # Break the loop if the user presses the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture and close any open windows
cap.release()
cv2.destroyAllWindows()