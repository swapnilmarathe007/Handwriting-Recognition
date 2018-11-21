import cv2
import numpy as np
# from image_helper_functions import *
#import image
image = cv2.imread('word_1.png')
#cv2.waitKey(0)

#grayscale
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imshow('gray',gray)
cv2.waitKey(0)

#binary
# ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
# cv2.imshow('second',thresh)
# cv2.waitKey(0)

#binary
ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
cv2.imshow('second',thresh)
cv2.waitKey(0)

#dilation
kernel = np.ones((5,5), np.uint8)
img_dilation = cv2.dilate(thresh, kernel, iterations=1)
cv2.imshow('dilated',img_dilation)
cv2.waitKey(0)

# thresh = cv2.dilate(thresh,kernel,iterations = 3)
# thresh = cv2.erode(thresh,kernel,iterations = 4)
# thresh = cv2.dilate(thresh,kernel,iterations = 2)
# thresh = cv2.erode(thresh,kernel,iterations = 3)


# thresh = cv2.dilate(thresh,None,iterations = 3)
# thresh = cv2.erode(thresh,None,iterations = 3)
# thresh = cv2.dilate(thresh,None,iterations = 4)
# thresh = cv2.erode(thresh,None,iterations = 6)
# # thresh = cv2.dilate(thresh,None,iterations = 1)

# thresh = cv2.erode(thresh,None,iterations = 1)
# thresh = cv2.dilate(thresh,None,iterations = 1)
# cv2.imshow('dilated',thresh)
# cv2.waitKey(0)

#find contours
im2,ctrs, hier = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#sort contours
sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

for i, ctr in enumerate(sorted_ctrs):
    # Get bounding box
    x, y, w, h = cv2.boundingRect(ctr)

    # Getting ROI
    roi = thresh[y:y+h, x:x+w]
    # # roi = 
    # gray = image_to_gray(roi)
    # #binary
    # thresh = gray_to_binary_inv(gray)
    cv2.imwrite("word_w_"+str(i)+".png" , roi)

    # show ROI
    cv2.imshow('segment no:'+str(i),roi)
    # cv2.rectangle(image,(x,y),( x + w, y + h ),(90,0,255),2)
    cv2.waitKey(0)

cv2.imshow('marked areas',image)
cv2.waitKey(0)
