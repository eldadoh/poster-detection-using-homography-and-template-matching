import os
import cv2
import glob 
import numpy as np 
from img_utils import plot_img_opencv,Normalize_float_binary_to_uint8_img,calc_image_range
from skimage.transform import resize as skimage_resize 
from template_matching import template_matching_func

def downsample_1_scene_according_1_template(scene_path,template_path,save = False , output_path = ''):

    scene_name = os.path.basename(scene_path)[:-len('.jpg')]
    template_name = os.path.basename(template_path)[:-len('.jpg')]
   

    scene = cv2.imread(scene_path,0)  
    template = cv2.imread(template_path,0) 
    
    h_w_template_array = np.array(template.shape[:2])
    scaling_ratios = np.linspace(start = 1,stop = 3,num = 10)
    
    for scale_ratio in scaling_ratios:
        
        h_scene_new , w_scene_new  = np.ceil(scale_ratio * h_w_template_array)
        
        downsampled_scene = skimage_resize(scene.copy(), ( h_scene_new, w_scene_new), anti_aliasing=True )
        downsampled_scene = Normalize_float_binary_to_uint8_img(downsampled_scene)
        
        scene_downsampled_name = f'{scene_name}_downsampled_to_{h_scene_new}_{w_scene_new}.jpg'
        scene_downsampled_name = os.path.join(output_path , scene_downsampled_name)

        if save : 
            cv2.imwrite(scene_downsampled_name,downsampled_scene)
    
    return downsampled_scene , template ,scene_downsampled_name , template_path

def main(): 


    scene_path = 'Resulotion_test_data/Second EXP - big letters Y U/scene_new_images/20210604_115029.jpg'
    template_path = '/home/arpalus/Work_Eldad/Arpalus_Code/Eldad-Local/arpalus-poster_detection/Resulotion_test_data/Second EXP - big letters Y U/input_images_hard_big_letters/APPBARBSBMN24x150421.jpg'
    downsampled_scene_dir_path = 'Resulotion_test_data/Second EXP - big letters Y U/scene_new_images/downsampled_scene_dir_path'
    downsampled_template_matching_results = 'Resulotion_test_data/Second EXP - big letters Y U/downsampled_template_matching_results'

    # scene_downsampled_img, template_img, scene_downsampled_img_path ,template_img_path = downsample_1_scene_according_1_template(scene_path,template_path,save = True , output_path = downsampled_scene_dir_path)
    
    for scene_downsampled_img_path in sorted(glob.glob(downsampled_scene_dir_path + '/*.jpg')):
    
        template_matching_func(scene_downsampled_img_path,template_path,output_path = downsampled_template_matching_results,show = True,save = True)

    



if __name__ == "__main__":

    main()
    