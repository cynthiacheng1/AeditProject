import cv2 
import PIL as pil
import tensorflow as tf
import torch 
import numpy as np
import pandas as pd


# print(f'OPENCV     = {cv2.__version__}')
# print(f'PILLOW     = {pil.__version__}')
# print(f'TENSORFLOW = {tf.__version__}')
# print(f'PYTORCH    = {torch.__version__}')
# print(f'NUMPY      = {np.__version__}')
# print(f'PANDAS     = {pd.__version__}')

# Use data provided here: https://drive.google.com/drive/folders/174XbPUnXnrCfzFlj2FqLx0_tczcbtpAx?usp=sharing
# Use these images to create a panomrama.

# CONCEPT 
# You need to use homographic morphing to stitch 2 images.
# Between 2 images, select at least 4 feature points common to both images.
# Find a homographic transformation matrix 'H' using the 4 feature points.
# Create a blank canvas - a black background large enough to hold images
# Using matric 'H', transfer one of the images to the space to the other image.
# Paste both images on the canvas.

# Points for steps:
# Stitch any two images - 10
# Create a stiched panomarama - 15


def draw_matches(img1, keypoints1, img2, keypoints2, matches):
  r, c = img1.shape[:2]
  r1, c1 = img2.shape[:2]

  # Create a blank image with the size of the first image + second image
  output_img = np.zeros((max([r, r1]), c+c1, 3), dtype='uint8')
  output_img[:r, :c, :] = np.dstack([img1, img1, img1])
  output_img[:r1, c:c+c1, :] = np.dstack([img2, img2, img2])

  # Go over all of the matching points and extract them
  for match in matches:
    img1_idx = match.queryIdx
    img2_idx = match.trainIdx
    (x1, y1) = keypoints1[img1_idx].pt
    (x2, y2) = keypoints2[img2_idx].pt

    # Draw circles on the keypoints
    cv2.circle(output_img, (int(x1),int(y1)), 4, (0, 255, 255), 1)
    cv2.circle(output_img, (int(x2)+c,int(y2)), 4, (0, 255, 255), 1)

    # Connect the same keypoints
    cv2.line(output_img, (int(x1),int(y1)), (int(x2)+c,int(y2)), (0, 255, 255), 1)
    
  return output_img

def warpImages(img1, img2, H):

  rows1, cols1 = img1.shape[:2]
  rows2, cols2 = img2.shape[:2]

  list_of_points_1 = np.float32([[0,0], [0, rows1],[cols1, rows1], [cols1, 0]]).reshape(-1, 1, 2)
  temp_points = np.float32([[0,0], [0,rows2], [cols2,rows2], [cols2,0]]).reshape(-1,1,2)

  # When we have established a homography we need to warp perspective
  # Change field of view
  list_of_points_2 = cv2.perspectiveTransform(temp_points, H)

  list_of_points = np.concatenate((list_of_points_1,list_of_points_2), axis=0)

  [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
  [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)
  
  translation_dist = [-x_min,-y_min]
  
  H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

  output_img = cv2.warpPerspective(img2, H_translation.dot(H), (x_max-x_min, y_max-y_min))
  output_img[translation_dist[1]:rows1+translation_dist[1], translation_dist[0]:cols1+translation_dist[0]] = img1

  return output_img


# Load our images

img1 = cv2.imread("pics/1/1.jpg")
img2 = cv2.imread("pics/1/2.jpg")
img3 = cv2.imread("pics/1/3.jpg")
img4 = cv2.imread("pics/1/4.jpg")
img5 = cv2.imread("pics/1/5.jpg")

img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
img3_gray = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
img4_gray = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)
img5_gray = cv2.cvtColor(img5, cv2.COLOR_BGR2GRAY)

# Create our ORB detector and detect keypoints and descriptors
orb = cv2.ORB_create(nfeatures=2000)

# Find the key points and descriptors with ORB
keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
keypoints2, descriptors2 = orb.detectAndCompute(img2, None)
keypoints3, descriptors3 = orb.detectAndCompute(img3, None)
keypoints4, descriptors4 = orb.detectAndCompute(img4, None)
keypoints5, descriptors5 = orb.detectAndCompute(img5, None)

# Create a BFMatcher object.
# It will find all of the matching keypoints on two images


def panoramicImage(descriptors1,descriptors2, keypoints1, keypoints2,img1,img2):
    bf = cv2.BFMatcher_create(cv2.NORM_HAMMING)

    # Find matching points
    matches = bf.knnMatch(descriptors1, descriptors2,k=2)

    #printing matched points 
    all_matches = []
    for m, n in matches:
      all_matches.append(m)
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    img3 = draw_matches(img1_gray, keypoints1, img2_gray, keypoints2, all_matches[:30])
    cv2.imshow('imagematched'+str(descriptors1[0][0]),img3)
    cv2.waitKey()
    
    # Finding the best matches
    good = []
    for m, n in matches:
        if m.distance < 0.6 * n.distance:
            good.append(m)

    # Set minimum match condition
    MIN_MATCH_COUNT = 10

    if len(good) > MIN_MATCH_COUNT:
        # Convert keypoints to an argument for findHomography
        src_pts = np.float32([ keypoints1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dst_pts = np.float32([ keypoints2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

        # Establish a homography
        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        
        res = warpImages(img2, img1, M)

        # cv2.imshow('finalresult',res)
        # cv2.waitKey()
        return res

pics_1_2 = panoramicImage(descriptors1,descriptors2,keypoints1,keypoints2,img1,img2)
cv2.imshow('pics_1_2',pics_1_2)
cv2.waitKey()

pics_2_3 = panoramicImage(descriptors2,descriptors3,keypoints2,keypoints3,img2,img3)
cv2.imshow('pics_2_3',pics_2_3)
cv2.waitKey()

pics_3_4 = panoramicImage(descriptors3,descriptors4,keypoints3,keypoints4,img3,img4)
cv2.imshow('pics_3_4',pics_3_4)
cv2.waitKey()

pics_4_5 = panoramicImage(descriptors4,descriptors5,keypoints4,keypoints5,img4,img5)
cv2.imshow('pics_4_5',pics_4_5)
cv2.waitKey()


# keypoints1_2, descriptors1_2 = orb.detectAndCompute(pics_1_2, None)
# keypoints2_3, descriptors2_3 = orb.detectAndCompute(pics_2_3, None)
# keypoints3_4, descriptors3_4 = orb.detectAndCompute(pics_3_4, None)

# pics_1_2_3 = panoramicImage(descriptors1_2,descriptors2_3, keypoints1_2, keypoints2_3, pics_1_2,pics_2_3)
# cv2.imshow('pics_1_2_3',pics_1_2_3)
# cv2.waitKey()

# pics_1_2_3 = panoramicImage(descriptors1_2,descriptors3, keypoints1_2, keypoints3, pics_1_2,img3)
# cv2.imshow('pics_1_2_3',pics_1_2_3)
# cv2.waitKey()


# pics_1_2_3_4 = panoramicImage(descriptors1_2,descriptors3_4, keypoints1_2, keypoints3_4, pics_1_2,pics_3_4)
# cv2.imshow('pics_1_2_3_4',pics_1_2_3_4)
# cv2.waitKey()

# pics_1_2_3 = panoramicImage(descriptors1_2,descriptors3, keypoints1_2, keypoints3, pics_1_2,img3)
# cv2.imshow('pics_1_2_3',pics_1_2_3)
# cv2.waitKey()




#reference code
# https://datahacker.rs/005-how-to-create-a-panorama-image-using-opencv-with-python/




