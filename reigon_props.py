import glob
import os
from skimage.measure import regionprops
import numpy as np 
import cv2 
from opencv_utils import Calc_hist_grayscale
from image_plots import plot_img_cv2

def Dilate(img,structuring=cv2.MORPH_RECT ,size = (3,3)):
    elem = cv2.getStructuringElement(structuring, size)
    dilated = cv2.dilate(img, elem)
    return dilated

def region_props(labeled_img,show = False):

    props = regionprops(labeled_img)

    if show : 
        for prop in props:
            print(prop, props[prop])

def main():
    
    img_path_android  = 'Resulotion_test_data/input_images/ANDROIDMN45x60218.jpg'
    img_path_apple = 'Resulotion_test_data/input_images/APPLEMN45x60218.jpg'
    OUTPUT_PATH = 'Resulotion_test_data/resized_images' 
    scene_test_image_android_apple = 'Resulotion_test_data/scene_test_images/IMG_1547_andriod_apple.jpg'

    plot_img_cv2(cv2.imread(img_path_android))
    # resize_image_multiple_scales(img_path_android, args = [1,2,4,8] ,output_path= OUTPUT_PATH)
    # resize_image_multiple_scales(img_path_apple, args = [1,2,4,8] ,output_path= OUTPUT_PATH)


if __name__ == "__main__" : 
    main() 
    pass