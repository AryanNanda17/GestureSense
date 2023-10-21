import argparse
import cv2 as cv
from imutils import paths
import imutils
# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--Path", required=True, help="Path to the image")
args = vars(parser.parse_args())

# Loop through images in specified folder
for imagePath in paths.list_images(args["Path"]):
	print (imagePath)
	image = cv.imread(imagePath)
	gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
	blurred = cv.GaussianBlur(gray, (7, 7), 0)
	#cv.imshow("Image", image)
	(T, Threshold) = cv.threshold(blurred, 15, 255, cv.THRESH_BINARY)
	#cv.imshow("Threshold Binary Inverse", threshInv)
	cnts = cv.findContours(Threshold.copy(), cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] 
	c = max(cnts, key=cv.contourArea)
	(x, y, w, h) = cv.boundingRect(c)
	crop = Threshold[y:y+h,x:x+w]
	cv.imwrite(imagePath,crop)