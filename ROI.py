#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 13:28:36 2019

@author: adithya+NegaMage+Vignesh
"""

from PIL import Image
import cv2
import numpy as np

im1 = cv2.imread("/home/adithya/Downloads/image1.jpg")
im_gray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
im_gray2 = cv2.GaussianBlur(im_gray1, (5, 5), 0)
ret, im_th = cv2.threshold(im_gray2, 90, 255, cv2.THRESH_BINARY_INV)


k=0
height=np.size(im_th, 0)
width=np.size(im_th, 1)
#we have th dimensions of height and width. We canfind what fraction of each row is filled with white pixels.
#and then we can choose those rows where the fraction is correspondingly accurate.
plottedx=[]
#plottedx is the list that will contain what fraction of each row is filled with white pixels. Remember that this is for the inverted image, not for the original image.

for x in range(height):
    k=0
    for y in range(width):
        k+=im_th[x][y]
    
    k/=height
    k/=255
    plottedx.append(k)

plottedy=[]
#plottedy is the equivalent in columns

for x in range(width):
    k=0
    for y in range(height):
        k+=im_th[y][x]
    
    k/=width
    k/=255
    plottedy.append(k)
    
#now we choose the part of plottedx and plottedy that  are greater than 0.0001 in the final testing image.
#for sake of computation we take it as 0.1 instead,
    
listx=[]
k=0
for x in range(height):
    
    if k==0 and plottedx[x]>0.1:
        listx.append(x)
        k=1
        continue
    elif k==1 and plottedx[x]==0.0:
        listx.append(x-1)
        k=0

listy=[]
k=0
for x in range(width):
    
    if k==0 and plottedy[x]>0.1:
        listy.append(x)
        k=1
        continue
    elif k==1 and plottedy[x]==0.0:
        listy.append(x-1)
        k=0
no_of_objects=len(listy)*len(listx)/4

#now, we can set up rectangular RoIs as follows:
#Top left x=listx[2i-1], y=listy[2i-1]
#Bottom right x=listx[2i], y=listy[2i]

#and i is the object number of a maximum of no_of_object objects.
#the rest is to shape these parts into images. and write trivial functions to trim further.
#and then we run the predictor!

for i in range(no_of_objects):
    rect = (listx[2*i-1], listy[2*i-1], listx[2*i],listy[2*i])
    #now, we have a bounding rectangle around the object, 
    #the image processing code goes here.




    
    
