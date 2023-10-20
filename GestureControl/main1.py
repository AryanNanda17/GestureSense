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

if __name__ == "__main__":
    capture = cv.VideoCapture(1)
    # wCam, hCam = 500,700 
    # capture.set(3, wCam)
    # capture.set(4, hCam)
    while(True):
        ret,frame = capture.read()

        cv.imshow('Hand Recognition', frame)
        prediction = predict(frame)
        print(prediction)
        cv.putText(frame,f'Hand Gesture: {prediction}', (40,70),cv.FONT_HERSHEY_SIMPLEX,3, (255,0,255),3)
        time.sleep(2)
        if cv.waitKey(20) & 0xFF == ord('d'):
            break

capture.release()
cv.destroyAllWindows()
