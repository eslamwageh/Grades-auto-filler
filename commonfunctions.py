
import cv2
import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np
from skimage.exposure import histogram
from matplotlib.pyplot import bar
from skimage.color import rgb2gray,rgb2hsv

# Convolution:
from scipy.signal import convolve2d
from scipy.signal.windows import gaussian
from scipy import fftpack
import math

from skimage.util import random_noise
from skimage.filters import median
from skimage.feature import canny



# Edges
from skimage.filters import sobel_h, sobel, sobel_v,roberts, prewitt

# Show the figures / plots inside the notebook
def show_images(images,titles=None):
    #This function is used to show image(s) with titles by sending an array of images and an array of associated titles.
    # images[0] will be drawn with the title titles[0] if exists
    # You aren't required to understand this function, use it as-is.
    n_ims = len(images)
    if titles is None: titles = ['(%d)' % i for i in range(1,n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image,title in zip(images,titles):
        a = fig.add_subplot(1,n_ims,n)
        if image.ndim == 2: 
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    plt.show() 


def showHist(img):
    # An "interface" to matplotlib.axes.Axes.hist() method
    plt.figure()
    imgHist = histogram(img, nbins=256)
    
    bar(imgHist[1].astype(np.uint8), imgHist[0], width=0.8, align='center')

def rectContour(contours):
    rectCon= []
    for i in contours:
        area= cv2.contourArea(i)
        # print("Area",area)
    

        if area>200:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i,0.04*peri, True)

            #print("Corner Points",len(approx))
            if len(approx) == 4:
                rectCon.append(i)
                rectCon = sorted(rectCon, key= cv2.contourArea,reverse=True)

    return rectCon  

def getCornerPoints(cont):
    peri = cv2.arcLength(cont, True)

    approx = cv2.approxPolyDP(cont, 0.04 * peri, True)

    return approx

def reorder(myPoints):
    myPoints = myPoints.reshape((4,2))
    myPointsNew = np.zeros((4,1,2), np.int32)
    add = myPoints.sum(1)
    #print(myPoints)
    #print(add)
    myPointsNew[0]= myPoints[np.argmin(add)] # [0, 0]
    myPointsNew[3]= myPoints[np.argmax(add)] # [w, h]
    # myPointsNew[3][0] -= 2
    diff = np.diff(myPoints,axis=1)
    myPointsNew[1]= myPoints[np.argmin(diff)] # [w, 0]
    # myPointsNew[1][0] -= 2
    myPointsNew[2]= myPoints[np.argmax(diff)] # [0, h]
    #print(diff)


    return myPointsNew

def getLowerBiggestContour(contours):
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Get the two largest contours
    if len(sorted_contours) >= 2:
        largest_two_contours = sorted_contours[:2]

        # Step 2: Compute the vertical position of each contour
        def get_vertical_position(contour):
            # Compute the bounding box and return the y-coordinate of the bottom edge
            _, _, _, y_bottom = cv2.boundingRect(contour)
            return y_bottom

        # Compare the two contours based on their vertical position
        contour_1, contour_2 = largest_two_contours
        if get_vertical_position(contour_1) > get_vertical_position(contour_2):
            largest_contour = contour_1
        else:
            largest_contour = contour_2
    else:
        # Fallback if less than 2 contours exist
        largest_contour = max(contours, key=cv2.contourArea) if contours else None
    return largest_contour