import shutil 
import itertools 
import os
import glob
import numpy as np
import cv2 
from cv2 import BFMatcher as bf
from matplotlib import pyplot as plt
from skimage.transform import resize as skimage_resize 
from image_plots import Plot_img_cv2 ,plots_opencv_images_pair_from_dir

def Calc_hist_grayscale(img , show) : 
    
    """
    Params:
    images : it is the source image of type uint8 or float32. it should be given in square brackets, ie, "[img]".
    channels : it is also given in square brackets. It is the index of channel for which we calculate histogram. For example, if input is grayscale image, its value is [0]. For color image, you can pass [0], [1] or [2] to calculate histogram of blue, green or red channel respectively.
    mask : mask image. To find histogram of full image, it is given as "None". But if you want to find histogram of particular region of image, you have to create a mask image for that and give it as mask. (I will show an example later.)
    histSize : this represents our BIN count. Need to be given in square brackets. For full scale, we pass [256].
    ranges : this is our RANGE. Normally, it is [0,256].
    """

    img = cv2.imread(img)
    img = cv2.cvtColor(img ,cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([img],[0],None,[256],[0,256])

    if show : 
        plt.hist(img.ravel(),256,[0,256]) 
        plt.show()

    return hist     

def Calc_hist_rgb(img , show) : 
    
    color = ('b','g','r')
    
    for i,col in enumerate(color):
        histr = cv2.calcHist([img],[i],None,[256],[0,256])
        plt.plot(histr,color = col)
        plt.xlim([0,256])
    plt.show()

def resize_image_to_multiple_scales(img_path, args ,output_path):
    
    """ 
        args = list of downsampling ratio numbers 
        ex : args = [2,4,8]
    """
    
    img = cv2.imread(img_path)

    h,w = img.shape[:2]

    for arg in args : 
        
        resized_img = skimage_resize(img.copy(), ( h//arg , w // arg ), anti_aliasing=True )
        
        resized_img*=255 #convert from float[0-1] to uint8 [0-255] 
        resized_img = resized_img.astype(np.uint8)
        
        h_,w_ = resized_img.shape[:2]

        resized_image_name = os.path.basename(img_path)[:-len('.jpg')] + '_factor' + str(arg*arg) + '_size' +f'{h_}' + '_' +f'{w_}' +'.jpg'
        resized_img_output_path  = os.path.join(output_path,resized_image_name)
        # Plot_img_cv2(resized_img,resize_flag=False)
        # cv2.imwrite(resized_img_output_path,resized_img)
    
def Calc_and_match_keypoints_between_two_images(img1_path, img2_path , show = False):
    
    img1_name = os.path.basename(img1_path)
    img2_name = os.path.basename(img2_path)

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


    sift = cv2.SIFT_create()
    kpA, desA = sift.detectAndCompute(gray1, None)
    kpB, desB = sift.detectAndCompute(gray2, None)

    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = bf.match(desB, desB)
    matches = sorted(matches, key=lambda x: x.distance)


    if show :

        matched_image = cv2.drawMatches(img1, kpA, img2, kpB, matches, None, flags=2)
        plt.imshow(cv2.cvtColor(matched_image, cv2.COLOR_BGR2RGB))
        plt.show() 

    return matches, kpA, desA,kpB, desB

def Find_global_min_and_max_in_single_chanel_array(array,mask = np.empty([])):

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



