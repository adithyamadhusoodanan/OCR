# Import the modules
from __future__ import print_function
import cv2
from sklearn.externals import joblib
from skimage.feature import hog
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn import datasets
from skimage import exposure
import imutils
from PIL import Image
import tensorflow as tf

# Load the classifier
clf = joblib.load("digits_cls.pkl")





i = 0

#read
im1 = cv2.imread("photo_2.jpg")
#im1 = Image.open("photo_1.jpg").convert('LA')
#im1 = np.asarray(im1)[:,:,0]

# grayscaling and gaussian filtering
im_gray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
im_gray2 = cv2.GaussianBlur(im_gray1, (5, 5), 0)

# Thresholding
ret, im_th = cv2.threshold(im_gray2, 90, 255, cv2.THRESH_BINARY_INV)

#finding contours
im2, ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
rects = [cv2.boundingRect(ctr) for ctr in ctrs]

#number of images
idx = 0

#bounding rectangles around digits
for ctr in ctrs:
    idx += 1
    x,y,w,h = cv2.boundingRect(ctr)
    roi=im2[y:y+h,x:x+w]
    
    #creating blank image of max dim    
    blank_image = np.zeros((max(w, h), max (w, h), 3), np.uint8)
    
    #for PIL
    cv2.imwrite('D:\\NITK\\Python\\crop\\temp\\' + 'roi.jpg', roi)
    cv2.imwrite('D:\\NITK\\Python\\crop\\temp\\' + 'blank.jpg', blank_image)
    
    #PIL Lazy Operation
    img1 = Image.open('D:\\NITK\\Python\\crop\\temp\\roi.jpg')
    img2 = Image.open('D:\\NITK\\Python\\crop\\temp\\blank.jpg')
    
    #pasting ROIs into blank for retaining aspect ratio
    img2.paste(img1, (int((max(w, h) - w) / 2), int((max(w, h) - h) / 2)))
    
    #coverting object into array
    img3 = np.asarray(img2)
    
    #resizing
    img4 = cv2.resize(img3, (20, 20), interpolation = cv2.INTER_AREA)
    BLACK = [0, 0, 0]
    img4 = cv2.copyMakeBorder(img4,4,4,4,4,cv2.BORDER_CONSTANT,value=BLACK)
    
    cv2.imwrite('D:\\NITK\\Python\\crop\\temp\\' + 'roi.jpg', img4)
    
    fim = Image.open("D:\\NITK\\Python\\crop\\temp\\roi.jpg").convert('LA')
    fim = np.asarray(fim)[:,:,0]
    
    fim.reshape((1, 784))
    print(fim.shape)
    
    #writetime
   # cv2.imwrite('D:\\NITK\\Python\\crop\\res\\' + str(idx) + '.jpg', fim)
    
    print(fim)
    
    fim = np.reshape(fim, (1,np.product(fim.shape)))
    
    
    # grab the image and classify it
    image = fim[[i]]
    #print(fim)
    prediction = clf.predict(image)[0]
    print (image.shape)
    # convert the image for a 64-dim array to an 8 x 8 image compatible with OpenCV,
    # then resize it to 32 x 32 pixels so we can see it better
    #image = image.reshape((8, 8)).astype("uint8")
    #image = exposure.rescale_intensity(image, out_range=(0, 255))
    #image = imutils.resize(image, width=32, inter=cv2.INTER_CUBIC)

    # show the prediction
    print("I think that digit is: {}".format(prediction))
    cv2.imshow("Image", img4)
    cv2.imwrite('D:\\NITK\\Python\\crop\\' + str(idx) + '.jpg', fim)
    cv2.waitKey(0)
    cv2.destroyAllWindows()