import cv2 as cv
import imutils
import numpy as np
import os
import uuid
import tensorflow as tf
from tensorflow import keras
import time
import pyautogui
from pynput.keyboard import Key, Controller

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
	
MODEL = tf.keras.models.load_model("/Users/Aryan/Documents/Projects/GestureSense/Models/5")
CLASS_NAMES = ['1finger','2finger','3finger','C','ThumbRight','fingersclosein','italydown','kitli','pinky','spreadoutpalm','yoyo']
def predict(img):

    gray_image = cv.resize(img,(128,128))
    # rgb_image = np.zeros((gray_image.shape[0], gray_image.shape[1], 3), dtype=np.uint8)
    # rgb_image[:, :, 0] = gray_image  
    # rgb_image[:, :, 1] = gray_image  
    # rgb_image[:, :, 2] = gray_image  
    img_batch = np.expand_dims(gray_image, 0)

    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return predicted_class

capture = cv.VideoCapture(1)
wCam, hCam = 640,480 
# pyautogui.failSafeCheck = False
screen_width, screen_height = pyautogui.size()
capture.set(3, wCam)
capture.set(4, hCam)
ROI = "10,350,225,590"
(top, right, bot, left) = np.int32(ROI.split(","))
md = MotionDetector()
numFrames = 0
k = 0
previous = "none"
consecutive = 0
mx = 0
my = 0
Centroid = 0
kernel_size = 5
kernel = np.ones((5, 5), np.uint8)
ptime = 0
fl1=0
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
	clone1 = None
	if numFrames < 32:
		md.update(gray)
	else:

		skin = md.detect(gray)
		if skin is not None:
			(thresh, c) = skin
			masked = cv.bitwise_and(hand, hand, mask=thresh)
			hsv = cv.cvtColor(masked, cv.COLOR_BGR2HSV)
			lower_range = np.array([0, 20, 70], dtype=np.uint8)
			upper_range = np.array([20, 255, 255], dtype=np.uint8)
			mask = cv.inRange(hsv, lower_range, upper_range)
			blurred_frame = cv.medianBlur(mask, kernel_size)
			erosion_result = cv.erode(blurred_frame, kernel, iterations=1)
			dilation_result = cv.dilate(erosion_result, kernel, iterations=1)
			# closing = cv.morphologyEx(dilation_result, cv.MORPH_CLOSE, kernel)
			# cv.imshow("blurred",blurred_frame)
			# cv.imshow("erosion result",erosion_result)
			cv.imshow("dilation_result",dilation_result)
			cv.imshow("hsv",mask)
			
			# cv.imshow("Mask", masked)

			cv.drawContours(clone, [c + (right, top)], -1, (0, 255, 0), 2)
			cv.imshow("Thresh", thresh)
			contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
			k = k+1
			if(numFrames>230):
				if len(contours) > 0:
					largest_contour = max(contours, key=cv.contourArea)
					Centroid = cv.moments(largest_contour)
				if (Centroid["m00"] != 0):
					mx = int(Centroid["m10"] / Centroid["m00"] + 1e-5)+350
					my =  int(Centroid["m01"] / Centroid["m00"] + 1e-5)+10
					cv.circle(clone, (mx,my), 5, (0, 0, 255), -1)
					print(str(mx) +" " +str(my))
				prediction = predict(masked)
				index = CLASS_NAMES.index(prediction)
				
				cv.putText(clone,str(prediction), (20,50),cv.FONT_HERSHEY_SIMPLEX,2, (255,0,255),2)
				if(previous==prediction):
					consecutive+=1
				if(consecutive>50):
					if(prediction=='2finger' or fl1==1):
						if(prediction=='spreadoutpalm'):
							fl1=0
						x1 = np.interp(mx,(right,wCam),(0,screen_width))
						y1 = np.interp(my,(top,hCam),(0,screen_height))
						pyautogui.moveTo(x1,y1)
						cv.circle(clone,(int(x1),int(y1)),5,(255,0,255),-1)
						if(prediction!='spreadoutpalm'):
							fl1=1
					elif(prediction=='kitli'):
						pyautogui.click(1100,200)
						pyautogui.hotkey('l')
					elif(prediction=='ThumbRight'):
						pyautogui.click(1100,200)
						pyautogui.hotkey('j')
					elif(prediction=='fingersclosein'):
						pyautogui.click(1100,500)
						pyautogui.hotkey('space')
					elif(prediction=='yoyo'):
						pyautogui.moveTo(300,300)
						pyautogui.hotkey('ctrl', 'right')
					elif(prediction=='1finger'):
						for _ in range(5):
							keyboard.press(Key.media_volume_up)
							keyboard.release(Key.media_volume_up)
							time.sleep(0.2)
					elif(prediction=='pinky'):
						for _ in range(5):
							keyboard.press(Key.media_volume_down)
							keyboard.release(Key.media_volume_down)
							time.sleep(0.2)
					elif(prediction=='C'):
						pyautogui.hotkey('f11')
					elif(prediction=='3finger'):
						pyautogui.click(1100,200)
						pyautogui.scroll(3)
					elif(prediction=='italydown'):
						pyautogui.click(1100,200)
						pyautogui.scroll(-3)
					consecutive = 0 
				previous = prediction
		cv.rectangle(clone, (left, top), (right, bot), (0, 0, 255), 2)
   
	numFrames += 1
	if numFrames >= 230:
		if fl ==1:
			print ("BackGround Subtraction Completed")
			fl=0
	else :
		print (numFrames)
		fl = 1
	ctime = time.time()
	fps = int(1/(ctime-ptime))
	cv.putText(clone,str(fps), (20,200),cv.FONT_HERSHEY_SIMPLEX,2, (255,0,255),2)
	ptime = ctime
	cv.imshow("Frame", clone)
	key = cv.waitKey(1) & 0xFF

	if key == ord("q"):
		break

capture.release()
cv.destroyAllWindows()
