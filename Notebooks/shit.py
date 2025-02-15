import cv2
import numpy as np

# Load the image
img = cv2.imread('../test-2.png')

# TO gray
gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)

#threshold Image
ret , thresh = cv2.threshold(gray , 127 , 255 , cv2.THRESH_BINARY)

# apply some dilation and erosion to join the gaps
thresh = cv2.dilate(thresh,None,iterations = 3)
thresh = cv2.erode(thresh,None,iterations = 2)

# Find the contours
img11 ,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

# For each contour, find the bounding rectangle and draw it
val = 0
for cnt in contours:
    x,y,w,h = cv2.boundingRect(cnt)
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    a = cv2.rectangle(thresh,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imshow('a',a)

# Finally show the image
cv2.imshow('img',img)
cv2.imshow('res',thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()