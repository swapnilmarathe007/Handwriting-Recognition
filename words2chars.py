from image_helper_functions import *

def words2chars(image_loc):
	try:
		# Read the image
		image = read_image(image_loc)
		# Turn to grayscale 
		gray = image_to_gray(image)
		# Turn to binary 
		thresh = gray_to_binary_inv(gray)
		# dilation
		kernel = np.ones((5,5), np.uint8)
		img_dilation = dilate_image(thresh,kernel,1)
		# Series of Dilation and Erosion to separate Characters 
		thresh = dilate_image(thresh,None,3)
		thresh = erode_image(thresh,None,6)
		thresh = dilate_image(thresh,None,4)
		thresh = erode_image(thresh,None,1)
		# thresh = dilate_image(thresh,None,5)
		# thresh = erode_image(thresh,None,1)
		#find contours
		sorted_ctrs = find_and_sort_contours(thresh)
		for i, ctr in enumerate(sorted_ctrs):
	    	# Get bounding box
			x, y, w, h = cv2.boundingRect(ctr)
		    # Getting ROI - Region of Interest
			print ("width",w)
			roi = image[y:y+h, x:x+w]
			cv2.imwrite("char_"+str(i)+".png",roi)
			# show ROI
			show_image("segment no "+str(i),roi)
			cv2.rectangle(image,(x,y),( x + w, y + h ),(90,0,255),2)
			cv2.waitKey(0)

		show_image('marked areas',image)
		cv2.waitKey(0)
	except Exception as error:
		print ("[-] ",error)


if __name__=="__main__":
	image_loc = "word_1.png"
	words2chars(image_loc)