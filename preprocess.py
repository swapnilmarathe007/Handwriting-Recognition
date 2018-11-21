#Import all necessary Packages
import cv2
import glob
import numpy as np 
folder_depth = 26
files = []
image_location = "/home/syedjafer/Documents/Handwriting_recognition_svm/projects/DATASET/"
try:
    # To get list of files in the current directory 
    for level in range(folder_depth):
        # Folders name start with 'a' till the folder depth 
        folder = chr(ord("a")+level)
        print (image_location+folder+"/")
        # To get the total number of files in individual folder  
        # label_len = len(glob.glob(image_location+folder+"/"+"*.png"))
        # List of all image files in all folders 
        files = files + glob.glob(image_location+folder+"/"+"*.png")
        # print(files) 
        # class Labels for all images : labels are the filename 
        # image_y = image_y + [folder] * (label_len)
except Exception as error : 
    print ("[-] ",error)

print (files)

for file in files:
	image = cv2.imread(file) # 0 for grayscale 2D array
	gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	ret , thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
	dilated_image = cv2.erode(thresh , None , iterations = 1)
	cv2.imwrite(file , dilated_image)	 
