import cv2
import imutils
import numpy as np
import os
import uuid
class MotionDetector:
	def __init__(self, accumWeight=0.5):
		self.accumWeight = accumWeight
		self.bg = None

	def update(self, image):
		if self.bg is None:
			self.bg = image.copy().astype("float")
			return
		cv2.accumulateWeighted(image, self.bg, self.accumWeight)

	def detect(self, image, tVal=25):
		delta = cv2.absdiff(self.bg.astype("uint8"), image)
		thresh = cv2.threshold(delta, tVal, 255, cv2.THRESH_BINARY)[1]
		cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		cnts = cnts[0] 
		if len(cnts) == 0:
			return None
		return (thresh, max(cnts, key=cv2.contourArea))
	
Images_Path = '/Users/Aryan/Documents/Projects/GestureSense/Create_Dataset'
label = 'pinky'
label_path = os.path.join(Images_Path, label)
os.makedirs(label_path, exist_ok=True)

camera = cv2.VideoCapture(1)
ROI = "10,350,225,590"
(top, right, bot, left) = np.int32(ROI.split(","))
md = MotionDetector()
numFrames = 0
k = 0
while True:

	(grabbed, frame) = camera.read()
	frame = imutils.resize(frame, width=600)
	frame = cv2.flip(frame, 1)
	clone = frame.copy()
	(frameH, frameW) = frame.shape[:2]
	roi = frame[top:bot, right:left]
	hand = roi.copy()
	gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (3, 3), 0)
	if numFrames < 32:
		md.update(gray)
	else:

		skin = md.detect(gray)
		if skin is not None:
			print('Collecting Images for {}'.format(label))
			(thresh, c) = skin
			masked = cv2.bitwise_and(hand, hand, mask=thresh)
			cv2.imshow("Mask", masked)
			name = os.path.join(label_path, '{}.jpg'.format(uuid.uuid1())) 
			cv2.imwrite(name,masked)
			cv2.drawContours(clone, [c + (right, top)], -1, (0, 255, 0), 2)
			cv2.imshow("Thresh", thresh)
			k = k+1
	cv2.rectangle(clone, (left, top), (right, bot), (0, 0, 255), 2)
	numFrames += 1
	if numFrames >= 230:
		if fl ==1:
			print ("BackGround Subtraction Completed")
			fl=0
	else :
		print (numFrames)
		fl = 1

	cv2.imshow("Frame", clone)
	key = cv2.waitKey(1) & 0xFF

	if key == ord("q"):
		break

camera.release()
cv2.destroyAllWindows()
