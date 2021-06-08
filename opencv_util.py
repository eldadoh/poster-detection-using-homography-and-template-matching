import numpy as np
import cv2 
from cv2 import BFMatcher as bf
from matplotlib import pyplot as plt
import os
import glob


def Blur(img, ker_size=(5, 5)):
    return cv2.GaussianBlur(img, ksize=ker_size, sigmaX=0)


def matchAB(img1_path, img2_path):
    
    img1_name = os.path.basename(img1_path)
    img2_name = os.path.basename(img2_path)

    imgA = cv2.imread(img1_path)
    imgB = cv2.imread(img2_path)

    
    grayA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)


    sift = cv2.SIFT_create()
    kpA, desA = sift.detectAndCompute(grayA, None)
    kpB, desB = sift.detectAndCompute(grayB, None)

    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desB, desB)
    matches = sorted(matches, key=lambda x: x.distance)
    matched_image = cv2.drawMatches(imgA, kpA, imgB, kpB, matches, None, flags=2)

    plt.imshow(cv2.cvtColor(matched_image, cv2.COLOR_BGR2RGB))
    plt.show() 

def find_global_min_and_max_in_single_chanel_array(array,mask = np.empty([])):

    """ 
        Finds the global minimum and maximum in an array.

        The function cv::minMaxLoc finds the minimum and maximum element values and their positions.
        The extremums are searched across the whole array or, 
        if mask is not an empty array, in the specified array region.

        The function do not work with multi-channel arrays. 
        If you need to find minimum or maximum elements across all the channels, 
        use Mat::reshape first to reinterpret the array as single-channel. 
        Or you may extract the particular channel using either extractImageCOI , or mixChannels , or split . 

        src	input single-channel array.
        minVal	pointer to the returned minimum value; NULL is used if not required.
        maxVal	pointer to the returned maximum value; NULL is used if not required.
        minLoc	pointer to the returned minimum location (in 2D case); NULL is used if not required.
        maxLoc	pointer to the returned maximum location (in 2D case); NULL is used if not required.
        mask	optional mask used to select a sub-array. 
    """ 
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(array, mask)



def knn_matcher(des1,des2):
    matches = bf.knnMatch(des1, des2, k=2) 
    good = []
    for (m1, m2) in matches: # for every descriptor, take closest two matches
        if m1.distance < 0.7 * m2.distance: # best match has to be this much closer than second best
            good.append(m1)



