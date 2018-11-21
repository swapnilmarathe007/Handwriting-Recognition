# Import all necessary Packages
import cv2 
import numpy as np
import matplotlib.pyplot as plt

image_loc = "testing_!.png"
img = cv2.imread(image_loc)
gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
#threshold Image
ret , thresh = cv2.threshold(gray , 127 , 255 , cv2.THRESH_BINARY)

#find contour 
image , contour , hier = cv2.findContours(thresh , cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)

val = 0
for c in contour:
    x,y,w,h = cv2.boundingRect(c)
    #draw a green rectangle 
    cv2.rectangle(img , (x,y) , (x+w , y+h) , (0,255,0) , 2)
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(img , [box] , 0 ,(0,0,255))
    # if (w > 50 and h > 60):
    #saving as images
#         if()
    roi = thresh[y:y+h, x:x+w]
    roi = cv2.resize(roi,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_AREA)
    print ("height",h,"width",w)
    h = h //2 
    w = w // 2
    print ("height",h,"width",w)
    if((100 - h) % 2 == 0):
        bordersize_top =  ( 100 -  (h) ) // 2
        bordersize_bottom = bordersize_top
        print ("HI ")
    else:
        bodersize_top = (( 100 -  (h) ) // 2) + 1
        bordersize_bottom = bordersize_top - 1 
        print("else")
    
    if((100 - w) % 2 == 0):
        bordersize_right = ( 100 -  (w) ) // 2
        bordersize_left = bordersize_right
    else:
        bordersize_right = (( 100 -  (w) ) // 2) + 1
        bordersize_left = bordersize_right - 1 
    

    print(bordersize_top , bordersize_bottom , bordersize_left , bordersize_right)
    mean = 255 
    try:
        ro = cv2.copyMakeBorder(roi, top=bordersize_top, bottom=bordersize_bottom, left=bordersize_left, right=bordersize_right, borderType= cv2.BORDER_CONSTANT, value=[mean,mean,mean] )
        filename = "roi_"+str(val)+".png"
        cv2.imwrite(filename , ro)
    except:
        pass
    plt.imshow(roi)
    
    val += 1

print(len(contour))
cv2.drawContours(img, contour, -1, (255, 255, 0), 1)
 
plt.imshow( img)
ESC = 27
