import numpy as np
import cv2 
import os
import shutil
from matplotlib import pyplot as plt
from image_plots import drawKeyPts_single_image , Plot_img_cv2 , plots_opencv_image_pair
from img_utils import Resize
from skimage.transform import resize as skimage_resize 

def resize_image_multiple_scales(img_path, args ,output_path):
    
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
        

def main():
    
    img_path_android  = 'Resulotion_test_data/input_images/ANDROIDMN45x60218.jpg'
    img_path_apple = 'Resulotion_test_data/input_images/APPLEMN45x60218.jpg'
    OUTPUT_PATH = 'Resulotion_test_data/resized_images' 
    scene_test_image_android_apple = 'Resulotion_test_data/scene_test_images/IMG_1547_andriod_apple.jpg'

    # resize_image_multiple_scales(img_path_android, args = [1,2,4,8] ,output_path= OUTPUT_PATH)
    # resize_image_multiple_scales(img_path_apple, args = [1,2,4,8] ,output_path= OUTPUT_PATH)


if __name__ == "__main__" : 
    # main() 
    pass