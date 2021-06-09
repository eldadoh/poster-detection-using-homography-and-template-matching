import glob
from img_utils import threshold_otsu
import os
from skimage.measure import regionprops
import numpy as np 
import cv2 
from opencv_utils import Calc_hist_grayscale
from image_plots import plot_img_cv2
from img_utils import dilate

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

    img = cv2.imread(img_path_android,0)
    th_img = threshold_otsu(img)
    dilate_img = dilate(th_img,size = (5,5) , iter = 10)

if __name__ == "__main__" : 
    main() 
    # pass