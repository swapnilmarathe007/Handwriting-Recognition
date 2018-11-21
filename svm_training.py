#Import all necessary Packages
import cv2
import numpy as np 
import glob 
from sklearn import svm 
from sklearn.externals import joblib
from sklearn import model_selection

def train_with_images(**kwargs):
    """
    
    Parameters  
    ----------
        image_location = Location of folder with collection of images
        folder_depth = Count of image folders inside image_location

    Requirements 
    ------------
        All the images in the image_location folder must be binary
        for better performance 

    Purpose 
    -------
        Collect the images from the folders and will convert them into 
        numpy arrays . Also these images will be labled with the actual
        value of that image .

    Directory Structure 
    -------------------
        image_location directory will comprise of folder_depth of folders.
        Each folder is been name with the class of the images inside that
        folder . 
        For Eg: Folder 'a' comprise of all images of alphabet letter 'a'
        these all will be in png format .

    """

    image_x = [] # Collection of all images
    image_y = [] # Collection of Labels 
    files = [] # Locations of all images in every folder 

    # Image Folders location and their depth
    image_location = kwargs["image_location"]
    folder_depth = kwargs["folder_depth"]

    try:
        # To get list of files in the current directory 
        for level in range(folder_depth):
            # Folders name start with 'a' till the folder depth 
            folder = chr(ord("a")+level)
            print (image_location+folder+"/")
            # To get the total number of files in individual folder  
            label_len = len(glob.glob(image_location+folder+"/"+"*.png"))
            # List of all image files in all folders 
            files = files + glob.glob(image_location+folder+"/"+"*.png")
            # print(files) 
            # class Labels for all images : labels are the filename 
            image_y = image_y + [folder] * (label_len)
    except Exception as error : 
        print ("[-] ",error)
        
    # Adding images to numpy array
    for file in files:
        image = cv2.imread(file,0) # 0 for grayscale 2D array
        image_x.append(image.flatten().tolist())

    # For Splitting of training data size
    test_size = 0.33
    seed = 78

    image_y = np.array(image_y)
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
                                image_x,
                                image_y ,
                                test_size=test_size,
                                random_state=seed)

    # Fit the model on 33%
    pkl_filename = "handwriting_svm.pkl"

    try:
        # Classifier 
        # kernel : linear for multiclass classification 
        clf = svm.SVC(kernel = 'linear', C = 1)
        clf.fit(x_train, y_train)
    except Exception as error:
        print ("[-] ",error)

    # Dump all the classified items using joblib 
    joblib.dump(clf, pkl_filename)

    # Score 
    # load the model from disk
    loaded_model = joblib.load(open(pkl_filename, 'rb'))
    result = loaded_model.score(x_test, y_test)
    print(result)

    # # To test the testing data 
    # try:
    #     for xx , yy in zip(x_test,y_test):
    #         print(clf.predict(np.array(xx).reshape(1,-1)),"=>",yy,"\n",)
    # except Exception as e:
    #     print ("[-] ",e)

def test_with_unseen_image(unseen_img_loc):
    # # To test with an unseen images 
    image = cv2.imread(unseen_image_loc,0)
    model = joblib.load(filename)
    test_image = np.array(image.flatten().tolist())
    print (model.predict(test_image.reshape(1,-1)) )

if __name__=="__main__":
    # Constants
    image_location = "/home/syedjafer/Documents/Handwriting_recognition_svm/projects/test1/images/"
    folder_depth = 8
    train_with_images(image_location=image_location,
                       folder_depth = folder_depth )
    # unseen_image_loc = "<image location>"
    # test_with_unseen_image(unseen_image_loc)
