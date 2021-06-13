import glob 
import os 
import sys
import cv2
import numpy as np
from skimage.transform import resize as skimage_resize 
from skimage.metrics import structural_similarity
from image_plots import plots_opencv_image_pair
from img_utils import create_dir_with_override, threshold_otsu
from img_utils import Blur,resize_img1_according_to_img2
from matplotlib import pyplot as plt
from img_utils import plot_img_opencv

def calc_ssim(poster,scene,show = False,ssim_gray = False) : 

    if poster.shape != scene.shape :
        if poster.shape[0] > scene.shape[0] and poster.shape[1] > scene.shape[1]:
            scene = skimage_resize(scene,(poster.shape[0] , poster.shape[1])) 
        else : 
            poster = skimage_resize(poster, (scene.shape[0] , scene.shape[1])) 
    
    
    score, diff = structural_similarity(scene, poster, multichannel=True, gaussian_weights=False, full=True)

    diff = diff.astype('float32') 


    if show :
        
        if ssim_gray: 
            
            diff = cv2.cvtColor(diff , cv2.COLOR_BGR2GRAY)
        
        plot_img_opencv(diff,resize_flag=True)

    

# find all matches of the template in the image
# returns an array of (x, y) coordinate of the top/left point of each match
# def getMatches(image, template, threshold):
#     result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
#     loc = np.where( result >= threshold)
#     results = zip(*loc[::-1])
#     return results
    
def find_maxima_points_on_corr_map_of_template_matching_above_th (img,template,th) : 
    result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where( result >= th)
    results = zip(*loc[::-1])
    return results
    
def template_matching_func(scene_path,template_path,output_path,show = False,save = False):

    """
        Input : scene image ,template 
        Returns : img + bbox if template match succed ,res ==> cross corr score map  
    """
    
    img_name = os.path.basename(scene_path)[:-len('.jpg')]
    template_name = os.path.basename(template_path)[:-len('.jpg')]

    img = cv2.imread(scene_path,0)
    template = cv2.imread(template_path,0)
    input_image = img.copy()
    
    w, h = template.shape[::-1]

    methods = ['cv2.TM_CCOEFF'] 
    # 'cv2.TM_CCOEFF', , 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

    for i,meth in enumerate(methods):
        output_img = input_image.copy()
        method = eval(meth)
        res = cv2.matchTemplate(output_img,template,method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc

        bottom_right = (top_left[0] + w, top_left[1] + h)

        cv2.rectangle(output_img,top_left, bottom_right, 0, 30)   
                        
        if show :
            
            # plot_img_opencv(img)
            # res = plot_img_opencv(res)
            pair_image_template_matching_result = plots_opencv_image_pair(template,img,show = True)
            pair_image_template_matching_result_ = plots_opencv_image_pair(res,img,show = True)

        if save:

            pair_image_template_matching_result = plots_opencv_image_pair(template,img,show = False)
            path = os.path.join(output_path ,img_name + '_' + f'{template_name}' +'_' +f'{meth}' +'.jpg')
            cv2.imwrite(path,pair_image_template_matching_result)
            # cv2.imwrite(path,cv2.cvtColor(pair_image_template_matching_result,cv2.COLOR_GRAY2BGR))

            pair_image_corr_map_score_matching_result_ = plots_opencv_image_pair(res,img,show = False)
            path = os.path.join(output_path ,img_name + '_' + f'{template_name}' +'_' +f'{meth}'+'_corr_map_score' +'.jpg')
            cv2.imwrite(path,pair_image_corr_map_score_matching_result_)
            # cv2.imwrite(path,cv2.cvtColor(pair_image_template_matching_result_,cv2.COLOR_GRAY2BGR))

    return output_img , res 


def template_matching_for_1_scene_several_templates_images(templates_dir_path,scene_image,output_path,show = False , save = False):
    """
        do template matching:
        - 1 scene image 
        - several templates 
    """
    
    # create_dir_with_override(output_path)

    for img in glob.glob(templates_dir_path + '/*.jpg'): 
        scene_image = (scene_image)
        template_matching_func(scene_image,img,output_path,save = True)
        

def main(): 

    poster_planogram_image = 'Data_new/planograms/planograms_parsed_images/APPBARBSDMN24x150421.jpg'
    templates_dir_path = 'Resulotion_test_data/Second EXP - big letters Y U/input_images_hard_big_letters'
    output_path = 'Resulotion_test_data/Second EXP - big letters Y U/template_matching_output'
    scene_dir_path = 'Resulotion_test_data/Second EXP - big letters Y U/scene_new_images'
    threshold_planograme_images_dir_path = 'Resulotion_test_data/Second EXP - big letters Y U/threshold_planograme_images_dir_path'
    template_matching_output_thresholded = 'Resulotion_test_data/Second EXP - big letters Y U/template_matching_output_thresholded'
    
    # template_matching_output_thresholded_blured = 'Resulotion_test_data/Second EXP - big letters Y U/template_matching_output_thresholded_blured'
    # #without threshold
    # for scene in (sorted(glob.glob(scene_dir_path + '/*.jpg'))):

    #     template_matching_for_1_scene_several_templates_images(templates_dir_path,scene,output_path,show = False , save = True)

    # for img in (sorted(glob.glob(templates_dir_path + '/*.jpg'))):
            
    #         threshold_img = threshold_otsu(img)
    #         threshold_img_name = os.path.join(threshold_planograme_images_dir_path,os.path.basename(img))
    #         cv2.imwrite(threshold_img_name, threshold_img)

    # for scene in (sorted(glob.glob(scene_dir_path + '/*.jpg'))):

    #     template_matching_for_1_scene_several_templates_images(templates_dir_path,scene,template_matching_output_thresholded,show = False , save = True)


    #   ########### scene_new_images_resized_according_to_template ##############

    """ 
        Note: result of this exp is : it doesnt work .
        we scale the scene to the template size 
        so only 1 pass is happening during the cross correltion of the template matching
        so we arent getting any corr map or satisfying result.
    """
    
    # scene_new_images_resized_according_to_template_dir_path = 'Resulotion_test_data/Second EXP - big letters Y U/scene_new_images_resized_according_to_template'
    # template_matching_exp3_scaling_the_scene_according_to_template_dir_path = 'Resulotion_test_data/Second EXP - big letters Y U/template_matching_exp3_scaling_the_scene_according_to_template_dir_path'
    
    # for scene in sorted(glob.glob(scene_dir_path + '/*.jpg')):

    #     for template in sorted(glob.glob(templates_dir_path + '/*.jpg')):

    #         _, __, scene_resized_img_path ,template_img_path = resize_img1_according_to_img2(scene , template ,output_dir_path=scene_new_images_resized_according_to_template_dir_path,save= True)
            
    #         template_matching_func(scene_resized_img_path,template_img_path,output_path = template_matching_exp3_scaling_the_scene_according_to_template_dir_path,show = True,save = True)

    #  #########################################################################

          ########### DEBUG ---- scene_new_images_resized_according_to_template ##############
      
    scene_new_images_resized_according_to_template_dir_path = 'Resulotion_test_data/Second EXP - big letters Y U/scene_new_images_resized_according_to_template'
    template_matching_exp3_scaling_the_scene_according_to_template_dir_path = 'Resulotion_test_data/Second EXP - big letters Y U/template_matching_exp3_scaling_the_scene_according_to_template_dir_path'
    
    #for debug
    templates_dir_path = 'Resulotion_test_data/Second EXP - big letters Y U/DEBUG_template_matching_exp3_scaling_the_scene_according_to_template_dir_path' 
    
    for scene in sorted(glob.glob(scene_dir_path + '/*.jpg')):
        
        #for debug
        scene = '20210604_115029_resized_according_to_APPBARBSBMN24x150421.jpg'

        for template in sorted(glob.glob(templates_dir_path + '/*.jpg')):

            _, __, scene_resized_img_path ,template_img_path = resize_img1_according_to_img2(scene , template ,output_dir_path=scene_new_images_resized_according_to_template_dir_path,save= True)
            
            template_matching_func(scene_resized_img_path,template_img_path,output_path = template_matching_exp3_scaling_the_scene_according_to_template_dir_path,show = True,save = True)

     #########################################################################
if __name__ == "__main__":

    main()
    