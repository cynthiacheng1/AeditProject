import cv2 
import PIL as pil
import tensorflow as tf
import torch 
import numpy as np
import pandas as pd


# Use data provided here: https://drive.google.com/drive/folders/1idvtHp12JY9YRxvb5Moatg9vj6ixM2T8?usp=sharing
# Use this image to count the number of:
# - RED stars: 4
# - YELLOW stars: 8
# - CYAN stars: 8
# - Total stars: 4 + 8 + 8 = 20

# CONCEPT

# Easier: WINDOW-BASED SEARCH
# Find 'C':  the color corresponding the element to be found.
# Create 'W': Window of size w x w, w = 1, 2, .. n, where n < width, n < height.
# Traverse over the image using the window size.
# Increment the count by one as as soon as:
# - Soft check: Majority of pixels of 'W' have the same color as 'C'
# - Hard check: All pixels of 'W' have the same color as 'C'

# Advanced: MORPHOLOGICAL SEARCH
# Find 'C':  the color corresponding the element to be found.
# You need to morphological operations, like erosion.
# Erode the spots that match the given color.

# NOTE: Advanced approach is more likely to get correct answer.

# Points for steps:
# Count of RED stars is correct - 4
# Count of YELLOW stars is correct - 8
# Count of CYAN stars is correct - 8
# Use advanced approach - Multiplier of 1.25.


# red = [(0,0,240),(10,10,255)] 
# red = [(159, 50, 70), (159, 240, 240)]
red = [(0, 50, 70), (9, 255, 255)]
cyan = [(110,50,50),(130,255,255)]
yellow = [(0,240,250),(10,255,255)]

dot_colors = [red, yellow, cyan]
colors = ["red", "yellow","cyan"]


img = cv2.imread("pics/2/starry_night.png")
# img = cv2.imread("pics/2/ex.png")


# apply medianBlur to smooth image before threshholding
blur= cv2.medianBlur(img, 7) # smooth image by 7x7 pixels, may need to adjust a bit
# cv2.imshow('blur',blur)
# cv2.waitKey()


for i in range(len(dot_colors)):
    output = img.copy()
    # apply threshhold color to white (255,255,255) and the rest to black (0,0,0)
    mask = cv2.inRange(blur,dot_colors[i][0],dot_colors[i][1]) 

    circles = cv2.HoughCircles(mask,cv2.HOUGH_GRADIENT,1,20,param1=20,param2=8,
                               minRadius=0,maxRadius=60)    

    index = 0
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")

        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # draw the circle in the output image, 
            # then draw a rectangle corresponding to the center of the circle
            cv2.circle(output, (x, y), r, (255, 0, 255), 2)
            cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (255, 0, 255), -1)

            index = index + 1
            print(str(index) + " : " + str(r) + ", (x,y) = " + str(x) + ', ' + str(y))
        print ('No. of {} circles detected = {}'.format(colors[i],index))
    else:
        print ('No. of {} circles detected = 0'.format(colors[i]))


#reference code 
#https://stackoverflow.com/questions/44439555/count-colored-dots-in-image