from image_helper_functions  import *

def img2sentence(image_loc):
    try:
        image = read_image(image_loc)
        #grayscale
        gray = image_to_gray(image)
        #binary
        thresh = gray_to_binary_inv(gray)
        #dilation
        kernel = np.ones((5,500), np.uint8)
        img_dilation = dilate_image(thresh, kernel,1)
        #find contours
        sorted_ctrs = find_and_sort_contours(thresh)
        for i, ctr in enumerate(sorted_ctrs):
            # Get bounding box
            x, y, w, h = cv2.boundingRect(ctr)

            # Getting ROI
            roi = image[y:y+h, x:x+w]
            cv2.imwrite("word_"+str(i)+".png",roi)
             # show ROI
            cv2.imshow('segment no:'+str(i),roi)
            cv2.rectangle(image,(x,y),( x + w, y + h ),(90,0,255),2)
            cv2.waitKey(0)
        show_image('Marked Areas',image)
        cv2.waitKey(0)
    except Exception as error:
        print ("[-] ",error)

if __name__ == "__main__":
    image_loc = "output.png"
    img2sentence(image_loc)