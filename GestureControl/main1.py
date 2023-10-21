import tensorflow as tf
from tensorflow import keras
import cv2 as cv
import numpy as np
import time
MODEL = tf.keras.models.load_model("/Users/Aryan/Documents/Projects/GestureSense/GestureControl/sign_language_mnist_model.h5")
CLASS_NAMES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25']
# CLASS_NAMES = ['1finger', '2finger', '3finger', '4finger', 'kitli', 
#               'neutral', 'pinch', 'pinky', 'snake', 'thumbsup', 'yoyo']

def predict(img):

    image = cv.resize(img,(28,28))
    image = image[:,:,:3]
    img_batch = np.expand_dims(image, 0)

    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    # return {d
    #     'class': predicted_class,
    #     # 'confidence': float(confidence)
    # }
    return predicted_class

if __name__ == "__main__":
    capture = cv.VideoCapture(0)
    print("HII")
    # wCam, hCam = 500,700 
    # capture.set(3, wCam)
    # capture.set(4, hCam)
    while(True):
        ret,frame = capture.read()
    
        print(frame.shape)
        prediction = predict(frame)
        print(prediction)
        cv.putText(frame,str(prediction), (20,50),cv.FONT_HERSHEY_SIMPLEX,3, (255,0,255),3)
        cv.imshow('Hand Recognition', frame)

        time.sleep(2)
        if cv.waitKey(20) & 0xFF == ord('d'):
            break

capture.release()
cv.destroyAllWindows()
