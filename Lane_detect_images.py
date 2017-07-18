# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 15:49:36 2017

@author: Toshiharu
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math

import glob, os
#import os
#for file in os.listdir("test_images"):
#    if file.endswith(".jpg"):
#        print(os.path.join("/test_images", file))
        

def init(imagesFolder):
    # Read in and grayscale the image
    os.chdir(imagesFolder)


def grayscale(img):
    # Grayscale transform
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def colorFilter(img,bgr,thresh):
    # Isolate certain color range of image
    # To isolate multiple colors, isolate them separately and bitwise added them
    # color is assumed in RGB format
    minBGR = np.array([bgr[0] - thresh, bgr[1] - thresh, bgr[2] - thresh])
    maxBGR = np.array([bgr[0] + thresh, bgr[1] + thresh, bgr[2] + thresh])
    return cv2.inRange(img,minBGR,maxBGR)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussianBlur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def roi(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    plt.figure()
    plt.imshow(image)
    plt.contour(mask,colors='b',linestyles='dashed')
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def Lane_lines_fit(houghLines,poly_degree=1):
    leftPoints  =[[],[]]
    rightPoints =[[],[]]
    leftCurve = np.zeros(shape=(1,2))
    rightCurve = np.zeros(shape=(1,2))
    
    for line in houghLines:
        for x1,y1,x2,y2 in line:
            #if m< 0, left line
            if ((y1-y2)/(x1-x2))<0:
                leftPoints=np.append(leftPoints,[[x1,x2],[y1,y2]],axis=1)
            else:
               rightPoints=np.append(rightPoints,[[x1,x2],[y1,y2]],axis=1)
    
#    print(rightPoints)
    leftCurve=np.polyfit(leftPoints[1,:],leftPoints[0,:],poly_degree)  #x=f(y)
    rightCurve=np.polyfit(rightPoints[1,:],rightPoints[0,:],poly_degree)  #x=f(y)

#    result = []
#    result=result.append(leftCurve)
#    result=result.append(rightCurve)
    
    return leftCurve,rightCurve

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
#    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    line_img = np.copy(image)*0
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_img, (x1, y1), (x2, y2), (255,0, 0), thickness=2)
## For debugging   
#    comb_img = cv2.addWeighted(image, 0.8, line_img, 1,0.5)
#    plt.figure()        
#    plt.imshow(comb_img)
#   draw_lines(line_img, lines)
    return lines

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

def drawLines(img,leftCurve,rightCurve,verLim):
    # Left line initial and end points
    imshape = img.shape
    y1_left = imshape[0]
    x1_left = int(y1_left*leftCurve[0]+leftCurve[1])
    
    y2_left = verLim
    x2_left = int(y2_left*leftCurve[0]+leftCurve[1])
    
    # Right line initial and end points
    y1_right = imshape[0]
    x1_right = int(y1_right*rightCurve[0]+rightCurve[1])
    
    y2_right = 350
    x2_right = int(y2_right*rightCurve[0]+rightCurve[1])
    
    fit_line_image = np.copy(img)*0
    
    cv2.line(fit_line_image,(x1_left,y1_left),(x2_left,y2_left),(255,0,0),20)
    cv2.line(fit_line_image,(x1_right,y1_right),(x2_right,y2_right),(255,0,0),20)

    fit_lines_edges = weighted_img(image,fit_line_image,0.8,1,0)
    plt.figure()
    plt.imshow(fit_lines_edges)

###############################################################################

imagesFolder = "test_images"
init(imagesFolder)  # Change dir to to imagesFolder
#
for file in glob.glob("*.jpg"):
    print(file)
    image = mpimg.imread(file)
#image = mpimg.imread('solidYellowCurve.jpg')

    imshape = image.shape
    gray = gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
        
    # Define a kernel size and apply Gaussian smoothing
    blur_gray = gaussianBlur(gray,kernel_size=5)
    
    # Define our parameters for Canny and apply
    low_threshold = 50
    high_threshold = 150
    edges = canny(blur_gray, low_threshold, high_threshold)
    
    # Masking Region of interest ()
    vertices = np.array([[(0,imshape[0]),(450, 300), (500,300), (imshape[1],imshape[0])]], dtype=np.int32)
    masked_edges = roi(edges,vertices)
    
    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 1 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 60  #1   # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 20 #5 #minimum number of pixels making up a line
    max_line_gap = 15 #1   # maximum gap in pixels between connectable line segments
    line_image = np.copy(image)*0 # creating a blank to draw lines on
    
    # Run Hough on edge detected image
    houghLines = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)
    #print(houghLines)
    
    # Identified the curve equation of each, left and right lanes
    leftCurve,rightCurve = Lane_lines_fit(houghLines,poly_degree=1)
    
    verLim = 350 # Vertical limit to draw the identified lane's curves
    drawLines(image,leftCurve,rightCurve,verLim)




