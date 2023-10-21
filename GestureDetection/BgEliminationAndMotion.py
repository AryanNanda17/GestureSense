import cv2 as cv
import imutils
import numpy as np
import os
import uuid
import tensorflow as tf
from tensorflow import keras
import time
class MotionDetector:
	def __init__(self, accumWeight=0.5):
		self.accumWeight = accumWeight
		self.bg = None

	def update(self, image):
		if self.bg is None:
			self.bg = image.copy().astype("float")
			return
		cv.accumulateWeighted(image, self.bg, self.accumWeight)

	def detect(self, image, tVal=25):
		delta = cv.absdiff(self.bg.astype("uint8"), image)
		thresh = cv.threshold(delta, tVal, 255, cv.THRESH_BINARY)[1]
		cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
		cnts = cnts[0] 
		if len(cnts) == 0:
			return None
		return (thresh, max(cnts, key=cv.contourArea))

MODEL = tf.keras.models.load_model("/Users/Aryan/Documents/Projects/GestureSense/Models/2")
CLASS_NAMES = ['1finger','2finger','3finger','4finger','Fist','fingersclosein',
			   'kitli','pinky','spreadoutpalm','thumbsdown','thumbsup','yoyo']

def predict(img):

    gray_image = cv.resize(img,(128,128))
    rgb_image = np.zeros((gray_image.shape[0], gray_image.shape[1], 3), dtype=np.uint8)
    rgb_image[:, :, 0] = gray_image  
    rgb_image[:, :, 1] = gray_image  
    rgb_image[:, :, 2] = gray_image  
    img_batch = np.expand_dims(rgb_image, 0)

    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return predicted_class

capture = cv.VideoCapture(1)
ROI = "10,350,225,590"
(top, right, bot, left) = np.int32(ROI.split(","))
md = MotionDetector()
numFrames = 0
k = 0
while True:
	
	(grabbed, frame) = capture.read()
	frame = imutils.resize(frame, width=600)
	frame = cv.flip(frame, 1)
	clone = frame.copy()
	(frameH, frameW) = frame.shape[:2]
	roi = frame[top:bot, right:left]
	hand = roi.copy()
	gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
	gray = cv.GaussianBlur(gray, (3, 3), 0)
	if numFrames < 32:
		md.update(gray)
	else:

		skin = md.detect(gray)
		if skin is not None:
			(thresh, c) = skin
			masked = cv.bitwise_and(hand, hand, mask=thresh)
			cv.imshow("Mask", masked)
			cv.drawContours(clone, [c + (right, top)], -1, (0, 255, 0), 2)
			cv.imshow("Thresh", thresh)
		
			k = k+1
			if(numFrames>230):
                
				prediction = predict(thresh)
				print(prediction)
				cv.putText(clone,str(prediction), (20,100),cv.FONT_HERSHEY_SIMPLEX,2, (255,0,255),1)
		cv.rectangle(clone, (left, top), (right, bot), (0, 0, 255), 2)
	numFrames += 1
	if numFrames >= 230:
		if fl ==1:
			print ("BackGround Subtraction Completed")
			fl=0
	else :
		print (numFrames)
		fl = 1
	cv.imshow("Frame", clone)
	key = cv.waitKey(1) & 0xFF

	if key == ord("q"):
		break

capture.release()
cv.destroyAllWindows()
