import cv2
from matplotlib.pyplot import plot
import numpy as np 
from img_utils import plot_img_matplotlib,plot_img_matplotlib,Normalize_img_by_min_max,plot_two_img_matplotlib
from skimage.transform import resize as skimage_resize 
from template_matching import custom_template_matching_func_for_production
import glob
import os 

def template_matching_couple_scales_scene_one_template(scene_path,template_path ,param_scale_ratio_low = 1.5 , param_scale_ratio_high = 3.5  ,  num_of_samples = 10,show = False,MIN_SCORE_TH = 0.75):

    max_correlation_score = 0
    final_number_of_clusters = 0 
    final_detection_coords = []
    final_detection_img = np.empty([])
   
    scene_rgb = cv2.imread(scene_path)  
    template_rgb = cv2.imread(template_path)

    scene = cv2.imread(scene_path,0)  
    template = cv2.imread(template_path,0) 
    
    h_w_template_array = np.array(template.shape[:2])
    scaling_ratios = np.linspace(param_scale_ratio_low ,param_scale_ratio_high,num_of_samples)
    
    for scale_ratio in scaling_ratios:
        
        h_scene_new , w_scene_new  = np.ceil(scale_ratio * h_w_template_array)
        
        downsampled_scene = skimage_resize(scene.copy(), ( h_scene_new, w_scene_new), anti_aliasing=True )
        downsampled_scene = Normalize_img_by_min_max(downsampled_scene)

        curr_correlation_score, detection_coords, detection_img,number_of_clusters  = custom_template_matching_func_for_production(downsampled_scene,template,Blur = True)
        
        if curr_correlation_score > max_correlation_score and curr_correlation_score > MIN_SCORE_TH : 

                max_correlation_score = curr_correlation_score
                final_detection_coords = detection_coords
                final_detection_img = detection_img
                final_number_of_clusters = number_of_clusters

    if max_correlation_score == 0 :

        print('---The template didnt found in the scene or found with not enough correlation score---')

        if show : 

            plot_two_img_matplotlib(template_rgb,scene_rgb,'\nNot found')

    else: 

        if show : 
            print('\n---There is a Detection---\n')    
            print(f'Correlation Score is : {max_correlation_score}')
            print(f'bbox coords are : {final_detection_coords}\n')
            print(f'number of clusters : {final_number_of_clusters}\n')
            plot_img_matplotlib(final_detection_img)

    return max_correlation_score,final_detection_coords,final_detection_img,final_number_of_clusters



def run_template_matching_on_various_templates(templates_dir_path,scene_path,show_ = False):
    
    min_number_of_clusters = np.inf 
    final_detection_image = np.empty([])
    final_bbox_coords = []
    final_correlation_score = 0
    final_template_name = ''

    for template_path in sorted(glob.glob(templates_dir_path +'/*.jpg')):

        curr_max_correlation_score,curr_detection_coords,curr_detection_img,curr_number_of_clusters = template_matching_couple_scales_scene_one_template(scene_path, template_path,show = False,MIN_SCORE_TH=0.5)
        
        curr_number_of_clusters = curr_number_of_clusters if curr_number_of_clusters is not None else min_number_of_clusters

        if curr_number_of_clusters < min_number_of_clusters :
            
            min_number_of_clusters = curr_number_of_clusters
            
            final_correlation_score = curr_max_correlation_score
            final_bbox_coords  = curr_detection_coords
            final_detection_image = curr_detection_img
            final_template_name = os.path.basename(template_path)[:-len('.jpg')]

    if show_ : 
        
        if final_correlation_score != 0:
            print(f'template name : {final_template_name}')
            print(f'final_correlation_score : {final_correlation_score}')
            print(f'final_bbox_coords : {final_bbox_coords}')
            
            if len(final_detection_image.shape) != 0 : 
                plot_img_matplotlib(final_detection_image)

    return final_correlation_score,final_bbox_coords,final_detection_image,final_template_name

def main(): 

    scene_path = 'Resulotion_test_data/Second EXP - big letters Y U/scene_new_images/20210604_115029.jpg'
    scene_path_O_ = 'Resulotion_test_data/Second EXP - big letters Y U/scene_new_images/20210604_115059.jpg'
    scene_path_only_O_ = 'Resulotion_test_data/Second EXP - big letters Y U/scene_new_images/20210604_115020.jpg'
    template_path = '/home/arpalus/Work_Eldad/Arpalus_Code/Eldad-Local/arpalus-poster_detection/Resulotion_test_data/Second EXP - big letters Y U/input_images_hard_big_letters/APPBARBSBMN24x150421.jpg'
    template_path_O_ = 'Data_new/planograms/resolution_test/Watchung/APPBARBSEMN39x270421.jpg'

    # template_matching_couple_scales_scene_one_template(scene_path_O_, template_path_O_)

    # FP , and gt_score < fp_score 
    #template_matching_couple_scales_scene_one_template(scene_path_only_O_, template_path,show = True,MIN_SCORE_TH=0.6) #0.8906000256538391
    #template_matching_couple_scales_scene_one_template(scene_path_only_O_, template_path_O_,show = True,MIN_SCORE_TH=0.6)
    
    # template_matching_couple_scales_scene_one_template(scene_path, template_path_O_,show = True,MIN_SCORE_TH=0.7)
    # template_matching_couple_scales_scene_one_template(scene_path, template_path,show = True,MIN_SCORE_TH=0.7)


#### one scene several templates #### 

    TEMPLATES_DIR_PATH = 'Resulotion_test_data/Second EXP - big letters Y U/input_images_hard_big_letters'
    
    # run_template_matching_on_various_templates(TEMPLATES_DIR_PATH,scene_path,show_=True)


#### all scene all template - final exp ####
    
    SCENE_IMAGES_DIR_PATH = 'Resulotion_test_data/Second EXP - big letters Y U/scene_new_images'
    for scene_path in sorted(glob.glob( SCENE_IMAGES_DIR_PATH + '/*.jpg')):

        scene_name = os.path.basename(scene_path)[:-len('.jpg')]
        print(f'\n--- current img : {scene_name} ---\n')
        run_template_matching_on_various_templates(TEMPLATES_DIR_PATH,scene_path,show_ = True)


if __name__ == "__main__":

    main()
    