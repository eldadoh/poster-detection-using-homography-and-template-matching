import numpy as np
import cv2 
import os
import shutil
import glob
from matplotlib import pyplot as plt
from img_utils import Blur,plot_img_opencv
from skimage.transform import resize as skimage_resize 

def resize_image_multiple_scales(img_path, args ,output_path,show = False , save = False):
    
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
        if show: 
            plot_img_opencv(resized_img,resize_flag=False)
        if save : 
            cv2.imwrite(resized_img_output_path,resized_img)
        

def main():
    
    img_path_android  = 'Resulotion_test_data/input_images/ANDROIDMN45x60218.jpg'
    img_path_apple = 'Resulotion_test_data/input_images/APPLEMN45x60218.jpg'
    OUTPUT_PATH = 'Resulotion_test_data/resized_images' 
    scene_test_image_android_apple = 'Resulotion_test_data/scene_test_images/IMG_1547_andriod_apple.jpg'

    android_better_resolution = 'Data_new/planograms/resolution_test/Watchung/ANDROIDMN13X30918.jpg'
    android_better_resolution_output = 'Resulotion_test_data/android_better_resolution_output'
    # resize_image_multiple_scales(img_path_android, args = [1,2,4,8] ,output_path= OUTPUT_PATH)
    # resize_image_multiple_scales(img_path_apple, args = [1,2,4,8] ,output_path= OUTPUT_PATH)
    # resize_image_multiple_scales(android_better_resolution, args = [1,2,4,8] ,output_path= android_better_resolution_output,save = True)
    # for img_path in sorted(glob.glob(f'{android_better_resolution_output}' + '/*.jpg')) :
    #     drawKeyPts_single_image(img_path,col = (0,0,255),th = 5 ,save = True,output_dir_path = android_better_resolution_output,return_key_points_count = True)
    
    
    KERNEL_ = (5,5)
    UPPER_Y_IMAGE_PATH = 'Resulotion_test_data/Second EXP - big letters Y U/input_images_hard_big_letters/APPBARBSAMN24x90421.jpg'
    UPPER_Y_RESIZED_DIR = 'Resulotion_test_data/Second EXP - big letters Y U/resized_upper_y_output'
    BLURED_IMAGE_PATH = 'Resulotion_test_data/Second EXP - big letters Y U/resized_upper_y_output/blured_img'+f'{KERNEL_}' +'.jpg'


    img = cv2.imread(UPPER_Y_IMAGE_PATH)
    blured_img = Blur(img,ker_size=KERNEL_)
    cv2.imwrite(BLURED_IMAGE_PATH, blured_img)
    resize_image_multiple_scales(BLURED_IMAGE_PATH, args = [1,2,4,8] ,output_path= UPPER_Y_RESIZED_DIR,save = True)
    
    
    # for scaled_img in sorted(glob.glob(UPPER_Y_RESIZED_DIR + '/*.jpg')):
    #     blured_scaled_img_name = os.path.basename(scaled_img)[:-len('.jpg')] + '_blured.jpg'
    #     blured_scaled_img = Blur(scaled_img,ker_size=(3,3))
    #     cv2.imwrite(blured_scaled_img_name,blured_scaled_img)

    
    # for img_path in sorted(glob.glob(f'{android_better_resolution_output}' + '/*.jpg')) :
    #     drawKeyPts_single_image(img_path,col = (0,0,255),th = 5 ,save = True,output_dir_path = android_better_resolution_output,return_key_points_count = True)
    
if __name__ == "__main__" : 
    main() 
    pass