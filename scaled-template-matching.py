import cv2
import numpy as np 
from img_utils import plot_img_matplotlib,plot_img_matplotlib,Normalize_img_by_min_max
from skimage.transform import resize as skimage_resize 
from template_matching import custom_template_matching_func_for_production

def template_matching_couple_scales_scene_one_template(scene_path,template_path ,param_scale_ratio_low = 1.5 , param_scale_ratio_high = 3  ,  num_of_samples = 6,show = False):

    max_correlation_score = 0
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

        curr_correlation_score, detection_coords, detection_img  = custom_template_matching_func_for_production(downsampled_scene,template,Blur = True)
        
        if curr_correlation_score > max_correlation_score : 
            
            max_correlation_score = curr_correlation_score
            final_detection_coords = detection_coords
            final_detection_img = detection_img

    if show : 

        print(max_correlation_score,final_detection_coords)
        plot_img_matplotlib(final_detection_img)

    return max_correlation_score,final_detection_coords,final_detection_img

def main(): 

    scene_path = 'Resulotion_test_data/Second EXP - big letters Y U/scene_new_images/20210604_115029.jpg'
    scene_path_O_ = 'Resulotion_test_data/Second EXP - big letters Y U/scene_new_images/20210604_115059.jpg'

    template_path = '/home/arpalus/Work_Eldad/Arpalus_Code/Eldad-Local/arpalus-poster_detection/Resulotion_test_data/Second EXP - big letters Y U/input_images_hard_big_letters/APPBARBSBMN24x150421.jpg'
    template_path_O_ = 'Data_new/planograms/resolution_test/Watchung/APPBARBSEMN39x270421.jpg'
    
    downsampled_scene_dir_path = 'Resulotion_test_data/Second EXP - big letters Y U/scene_new_images/downsampled_scene_dir_path_Y'
    downsampled_scene_dir_path_O_='Resulotion_test_data/Second EXP - big letters Y U/scene_new_images/downsampled_scene_dir_path_O'
    
    downsampled_template_matching_results = 'Resulotion_test_data/Second EXP - big letters Y U/downsampled_template_matching_results_hard_postive'
    downsampled_template_matching_results_O_ = 'Resulotion_test_data/Second EXP - big letters Y U/downsampled_template_matching_results_easy_postive'

    template_matching_couple_scales_scene_one_template(scene_path, template_path,show = True)
    # template_matching_couple_scales_scene_one_template(scene_path_O_, template_path_O_)

if __name__ == "__main__":

    main()
    