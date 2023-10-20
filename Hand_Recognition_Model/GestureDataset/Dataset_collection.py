import cv2 as cv
import time
import os
import uuid
# , '2finger', '3finger', '4finger', 'kitli', 'pinch', 'pinky', 'snake', 'thumbsup', 'yoyo'
Images_Path = '/Users/Aryan/Documents/Projects/DataSetCollection/GestureDataSet/CollectedImages'
labels = ['pinky', 'snake']
number_of_images_per_gesture = 100 

for label in labels:
    label_path = os.path.join(Images_Path, label)
    os.makedirs(label_path, exist_ok=True) 
    capture = cv.VideoCapture(1) 
    print('Collected Images for {}'.format(label))
    time.sleep(5)  

    for imgnum in range(number_of_images_per_gesture):
        ret, frame = capture.read()
        if not ret:
            print("Failed to capture a frame")
            break
        imagename = os.path.join(label_path, '{}.jpg'.format(uuid.uuid1()))  
        cv.imwrite(imagename, frame)  
        cv.imshow('frame', frame)  
        time.sleep(1)  
        if cv.waitKey(1) & 0xFF == ord('q'):  
            break
    capture.release()
    cv.destroyAllWindows()  
