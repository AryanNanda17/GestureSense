import tensorflow as tf
import cv2 as cv
import numpy as np
import time

MODEL = tf.keras.models.load_model("C:/Users/mihir/Desktop/sign_language_mnist_model.h5")

CLASS_NAMES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25']

def predict(img):
    image = cv.resize(img, (28, 28))
    image = image[:,:,:3]
    img_batch = np.expand_dims(image, 0)
    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return predicted_class

if __name__ == "__main":
    capture = cv.VideoCapture(0)
    while True:
        ret, frame = capture.read()
        
        if not ret:
            print("Failed to capture a frame")
            continue
        
        cv.imshow('Hand Recognition', frame)
        
        prediction = predict(frame)
        print(prediction)
        cv.putText(frame, f'Hand Gesture: {prediction}', (40, 70), cv.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 3)
        
        if cv.waitKey(20) & 0xFF == ord('d'):
            break

    capture.release()
    cv.destroyAllWindows()
