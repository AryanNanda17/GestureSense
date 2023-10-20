import cv2 as cv 
import numpy as np
import mediapipe as mp
import time

captureture = cv.VideoCapture(1)

mpHands = mp.solutions.hands
Hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
while True:
    ret, frame = captureture.read()

    ImgRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = Hands.process(ImgRGB)
    if results.multi_hand_landmarks:
        for handlms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(frame, handlms, mpHands.HAND_CONNECTIONS)
            lm_list = []
            for lm in handlms.landmark:
                h, w, c = frame.shape
                x, y = int(lm.x * w), int(lm.y * h)
                lm_list.append((x, y))

            landmarks_np = np.array(lm_list, dtype=np.int32)

            hull = cv.convexHull(landmarks_np)

            x, y, w, h = cv.boundingRect(hull)

            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv.imshow("image", frame)
    if cv.waitKey(20) & 0xFF == ord('d'):
        break

captureture.release()
cv.destroyAllWindows()