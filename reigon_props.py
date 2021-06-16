import numpy as np 
import cv2 
import glob
import os
from opencv_utils import Calc_hist_grayscale
from image_plots import plot_img_opencv
from img_utils import dilate,threshold_otsu,calc_image_range
from skimage.measure import label,regionprops,regionprops_table

"""
    info about regionprops method of skimage:
    1.label
        https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.label
    2.regionprops
        https://scikit-image.org/docs/dev/api/skimage.measure.html#regionprops  
 """ 

def apply_regionprops(img_path,dilate_flag = False, display= 'Regular'):

    """
       each crop is like a cluster
       choose properties is display is 'pandas_table (use the linke above)
    '"""

    img = cv2.imread(img_path,0)
    
    th_img = threshold_otsu(img)
    
    if dilate_flag:
        dilate_img = dilate(th_img,size = (5,5) , iter = 10)
    
    labeled_img = label(dilate_img,connectivity=2)
    
    props=regionprops(labeled_img)
    
    if display == 'Regular':

        for prop in props:
            
            print('Label: {} >> Object size: {}'.format(prop.label, prop.area))
            print('Label: {} >> Object size: {}'.format(prop.label, prop.bbox_area))

    elif display == 'pandas_table':

        table_props = regionprops_table(labeled_img, img.copy(),properties=('centroid',
                                                 'orientation',
                                                 'major_axis_length',
                                                 'minor_axis_length'))
        
        print(table_props)

def main() : 

    img_path_android  = 'Resulotion_test_data/input_images/ANDROIDMN45x60218.jpg'
    img_path_apple = 'Resulotion_test_data/input_images/APPLEMN45x60218.jpg'
    OUTPUT_PATH = 'Resulotion_test_data/resized_images' 
    scene_test_image_android_apple = 'Resulotion_test_data/scene_test_images/IMG_1547_andriod_apple.jpg'

    img_path = 'Figure_2.png'
    apply_regionprops(img_path,dilate_flag = True, display= 'Regular')
    apply_regionprops(img_path,dilate_flag = True, display= 'pandas_table')


if __name__ == "__main__" : 
    main() 
    # pass  