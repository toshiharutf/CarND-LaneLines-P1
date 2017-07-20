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
        
def grayscale(img):
    # Grayscale transform
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def colorFilter(img,lowerColor,upperColor):
    """
    Isolate certain color range of image
    To isolate multiple colors, isolate them separately and bitwise added them
    color is assumed in RGB format
    """
    return cv2.inRange(img,lowerColor,upperColor)

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
    """defining a blank mask to start with"""
    mask = np.zeros_like(img)   
    
    """defining a 3 channel or 1 channel color to fill the mask with depending
    on the input image
    """
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    """filling pixels inside the polygon defined by "vertices" with the
    fill color    
    """
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    plt.figure()
    plt.imshow(image)
    plt.contour(mask,colors='b',linestyles='dashed')
    
    """returning the image only where mask pixels are nonzero"""
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def Lane_lines_fit(houghLines,poly_degree=1):
    """
    The points obtained from the Hough transformation are used to
    identify the polynomial curve parameters using the function polyfit
    The outputs of the Lane_lines_fit function are the left and right
    curve 
    """
    leftPoints  =[[],[]]
    rightPoints =[[],[]]
    leftCurve = np.zeros(shape=(1,2))
    rightCurve = np.zeros(shape=(1,2))
    
    for line in houghLines:
        for x1,y1,x2,y2 in line:
            #if m< 0, left line
            m = (y1-y2)/(x1-x2)
#            print(m)
            if m<-0.5:
                leftPoints=np.append(leftPoints,[[x1,x2],[y1,y2]],axis=1)
            elif m>0.5:
               rightPoints=np.append(rightPoints,[[x1,x2],[y1,y2]],axis=1)
    
#    print(rightPoints)
    leftCurve=np.polyfit(leftPoints[1,:],leftPoints[0,:],poly_degree)  #    x=f(y)
    rightCurve=np.polyfit(rightPoints[1,:],rightPoints[0,:],poly_degree)  # x=f(y)
   
    return leftCurve,rightCurve

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
#    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    line_img = np.copy(img)*0
    
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
    """
    This function takes an image, and the curve equation of the left and right
    lanes. Then, it draws this two curves over the image. 

    By default, both equation start at the bottom of the figure and finish at
    y = verLim
    """
    """ Left line initial and end points"""
    imshape = img.shape
    y1_left = imshape[0]
    x1_left = int(y1_left*leftCurve[0]+leftCurve[1])
    
    y2_left = verLim
    x2_left = int(y2_left*leftCurve[0]+leftCurve[1])
    
    """ Right line initial and end points """
    y1_right = imshape[0]
    x1_right = int(y1_right*rightCurve[0]+rightCurve[1])
    
    y2_right = verLim
    x2_right = int(y2_right*rightCurve[0]+rightCurve[1])
    
    fit_line_image = np.copy(img)*0
    
    """ Drawing the curves """
    cv2.line(fit_line_image,(x1_left,y1_left),(x2_left,y2_left),(0,0,255),10)
    cv2.line(fit_line_image,(x1_right,y1_right),(x2_right,y2_right),(0,0,255),10)
    
    """
    Overlaying the curves on the input image, applying a previous transparency
    transformation
    """
    return  weighted_img(image,fit_line_image,0.8,1,0)


###############################################################################

#imagesFolder = "test_images"
imagesFolder = "images_challenge"
outputFolder = "images_output"
#init(imagesFolder)  # Change dir to to imagesFolder
#

for file in os.listdir(imagesFolder):
    if file.endswith(".jpg"):
#        image = mpimg.imread(imagesFolder+"/"+file)
        image = cv2.imread(imagesFolder+"/"+file)
        
        #Filter white color  in BGR order!!!
        lowerWhite = np.array([195,195,195])
        upperWhite = np.array([255,255,255])
  
#        lowerYellow = np.array([80, 190, 215])
#        upperYellow = np.array([150, 255, 255]) 

        """Challenge colors"""
#        lowerYellow = np.array([30, 180, 215])
#        upperYellow = np.array([150, 255, 255]) 
        lowerYellow = np.array([15, 170, 215])
        upperYellow = np.array([160, 255, 255]) 
        
        imageWhites = colorFilter(image,lowerWhite,upperWhite)
        imageYellows = colorFilter(image,lowerYellow,upperYellow)
        imageFiltered = cv2.bitwise_or(imageWhites, imageYellows)
        
        plt.figure()
        plt.imshow(imageFiltered)
    #for file in glob.glob("*.jpg"):
    #    print(file)
    #    image = mpimg.imread(file)
    #image = mpimg.imread('solidYellowCurve.jpg')
    
        """ Masking Region of interest"""
        imshape = image.shape
        #        vertices = np.array([[(0,imshape[0]),(450, 300), (500,300), (imshape[1],imshape[0])]], dtype=np.int32)
        """ Challenge"""
        vertices = np.array([[(0,660),(550, 450), (730,450), (imshape[1],660)]], dtype=np.int32)
        maskedImg = roi(imageFiltered,vertices)
    
#        gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
            
        """ Define a kernel size and apply Gaussian smoothing"""
        blur_gray = gaussianBlur(maskedImg,kernel_size=5)
        
        # Define our parameters for Canny and apply
        low_threshold = 50
        high_threshold = 200
        edges = canny(blur_gray, low_threshold, high_threshold)
#        plt.figure()
#        plt.imshow(edges)
        

        
        """
        Define the Hough transform parameters
        Make a blank the same size as our image to draw on
        """
        rho = 1 # distance resolution in pixels of the Hough grid
        theta = np.pi/180 # angular resolution in radians of the Hough grid
        threshold =30  #1   # minimum number of votes (intersections in Hough grid cell)
        min_line_length = 20 #5 #minimum number of pixels making up a line
        max_line_gap = 15 #1   # maximum gap in pixels between connectable line segments
        line_image = np.copy(image)*0 # creating a blank to draw lines on
        
        """ Run Hough on edge detected image"""
        houghLines = hough_lines(edges, rho, theta, threshold, min_line_length, max_line_gap)
        #print(houghLines)
        
        """ Identified the curve equation of each, left and right lanes"""
        leftCurve,rightCurve = Lane_lines_fit(houghLines,poly_degree=1)
        
#        verLim = 350 # Vertical limit to draw the identified lane's curves
        """Challenge verLim"""
        verLim = 450
        
        output_img = drawLines(image,leftCurve,rightCurve,verLim)
        
        
        cv2.imwrite(outputFolder+"/"+file[:-4]+'-output.jpg',output_img)
    
    #print(file)
        plt.figure()
        plt.title(file[:-4]+'-output')
        output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)  #change to RGB
        plt.imshow(output_img)
    




