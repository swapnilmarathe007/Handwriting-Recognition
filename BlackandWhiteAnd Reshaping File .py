# Import all necessary packages
import cv2 
import numpy as np
import matplotlib.pyplot as plt

# Converting to Binary 

image_location = "/home/syedjafer/Documents/Handwriting_recognition_svm/images/a/a_00.png"
img = cv2.imread(image_location,0) 
(thresh, im_bw) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
cv2.imwrite('bw_image.png', im_bw)

# Reshaping Dimensions of Image
dim = (100 , 100)
resized = cv2.resize(im_bw, dim, interpolation = cv2.INTER_AREA)
plt.imshow(resized)
cv2.imwrite('bw_image.png', resized)
