# Import necessary packages
import cv2
import numpy as np
"""

Image helper functions comprises of all the basic functions 
that are been used towards this project .  

"""

def read_image(image_loc):
	image = cv2.imread(image_loc)
	return image

def image_to_gray(image):
	gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	return gray

def gray_to_binary(gray):
	ret , thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
	return thresh

def gray_to_binary_inv(gray):
	ret , thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
	return thresh

def show_image(display_name, image):
	cv2.imshow(display_name,image)
	cv2.waitKey(0)

def dilate_image(thresh, kernel, iterations):
	dilated_image = cv2.dilate(thresh , kernel , iterations = iterations)
	return dilated_image

def erode_image(thresh , kernel , iterations):
	eroded_image = cv2.erode(thresh , kernel , iterations = iterations)
	return eroded_image 

def find_and_sort_contours(thresh):
	#find contours
	image ,ctrs, hier = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	#sort contours
	sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
	return sorted_ctrs
