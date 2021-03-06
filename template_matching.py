import numpy as np
import glob 
import os 
import cv2
from matplotlib import pyplot as plt
from skimage.transform import resize as skimage_resize 
from skimage.metrics import structural_similarity
from image_plots import plots_opencv_image_pair
from img_utils import plot_img_opencv , plot_img_matplotlib, plot_two_img_matplotlib
from reigon_props import regionprops_criteria_for_template_matching 

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
        
    return score, diff

def template_matching_func(scene_path,template_path,output_path,show = False,save = False,Blur = True):

    """
        Template Matching using gray-scale scene and template 
        Input : rgb scene image ,rgb template 
        Returns : img + bbox if template match succed ,res ==>  1d cross corr score map  

            max_score_of_current_img score of current scene scale
            detection_coords : list:[top left ,bottom right]
            img_rgb: the img with bbox around detection 

        # 'cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED'
    """
    
    img_name = os.path.basename(scene_path)[:-len('.jpg')]
    template_name = os.path.basename(template_path)[:-len('.jpg')]

    img_rgb = cv2.imread(scene_path)
    template_rgb = cv2.imread(template_path)

    img = cv2.imread(scene_path,0) 
    template = cv2.imread(template_path,0)
    
    w, h = template.shape[::-1]

    methods = ['cv2.TM_CCORR_NORMED']
    CORR_MAP_RES_TH = 0.5
    BLUR_KER_SIZE = 50 
        

    for i,meth in enumerate(methods):
        
        output_img = img.copy()
        method = eval(meth)
        
        try:  # in case of couple templates in one scene 
               
            res = cv2.matchTemplate(output_img,template,method)
            loc = np.where( res >= CORR_MAP_RES_TH) #return [coord_y_arr , coord_x_arr]
            res = zip(*loc[::-1])      #fliping to [coord_x_arr , coord_y_arr]
            min_val, max_score_of_current_img, min_loc, max_loc = cv2.minMaxLoc(res) #gets only single changel array
            
            print(f'Achieved threshold of {CORR_MAP_RES_TH} !')
            print(f'Coord of the max_loc is : {max_loc} with Score of {max_score_of_current_img}')
            
        
        except Exception as e : #assuming only single template object in the scene 

            res = cv2.matchTemplate(output_img,template,method)

            min_val, max_score_of_current_img, min_loc, max_loc = cv2.minMaxLoc(res)
           
            # print('Score before low pass')
            # print(f'the coord of the max_loc is : {max_loc} with score of {max_score_of_current_img}')

            if Blur: 
            
                res = cv2.blur(res,(BLUR_KER_SIZE,BLUR_KER_SIZE), 0)

                min_val, max_score_of_current_img, min_loc, max_loc = cv2.minMaxLoc(res)

                # print('Score after low pass')
                # print(f'the coord of the max_loc is : {max_loc} with score of {max_score_of_current_img}')

        
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc

        bottom_right = (top_left[0] + w, top_left[1] + h)
        
        detection_coords = [top_left,bottom_right]

        cv2.circle(res, top_left, radius=1, color=(0, 255, 0), thickness=10) #plot max_loc point
        cv2.rectangle(img_rgb,top_left, bottom_right, 0, 30)                 #plot bbox around detection

        if show :  # res range is [0-1]
            
            plt.imshow(res, cmap='hot', interpolation ='nearest')
            plt.show()
            plt.close()

            pair_image_template_matching_result = plots_opencv_image_pair(template_rgb,img_rgb,show = True)
            

        if save:
            
            pair_image_template_matching_result = plots_opencv_image_pair(template_rgb,img_rgb,show = False)
            path = os.path.join(output_path ,img_name + '_' + f'{template_name}' +'_' +f'{meth}' +'.jpg')
            cv2.imwrite(path,pair_image_template_matching_result)

            pair_image_corr_map_score_matching_result_ = plots_opencv_image_pair(res,img_rgb,show = False)
            path = os.path.join(output_path ,img_name + '_' + f'{template_name}' +'_' +f'{meth}'+'_corr_map_score' +'.jpg')
            cv2.imwrite(path,pair_image_corr_map_score_matching_result_)

    return max_score_of_current_img, detection_coords, pair_image_template_matching_result 
            

def template_matching_one_scene_several_templates(templates_dir_path,scene_image,output_path,show = False , save = False):
    """
        do template matching:
        - 1 scene image 
        - several templates 
    """
    
    # create_dir_with_override(output_path)

    for img in glob.glob(templates_dir_path + '/*.jpg'): 
        scene_image = (scene_image)
        template_matching_func(scene_image,img,output_path,save = True)
    

def analyze_correlation_map(corr_map_img,PARAM_SEGMENTATION_BY_COLOR_TH):
    corr_map_thresholded = np.where(corr_map_img >= np.max(corr_map_img) - PARAM_SEGMENTATION_BY_COLOR_TH , 1, 0 ) # for visualizations 
    # plot_img_matplotlib(corr_map_thresholded)
    labeled_img, num_of_clusters  = regionprops_criteria_for_template_matching(corr_map_img)
    # plot_img_matplotlib(labeled_img)

    return labeled_img,num_of_clusters

def custom_template_matching_func_for_production(scene,template,Blur = True,CORR_MAP_RES_TH=0.75,PARAM_SEGMENTATION_BY_COLOR_TH = 0.1):

    
    img = scene
    template = template 

    w, h = template.shape[::-1]

    methods = ['cv2.TM_CCORR_NORMED']

    BLUR_KER_SIZE = 50 
        
    for i,meth in enumerate(methods):
        
        output_img = img.copy()
        method = eval(meth)
        
        try:  # in case of couple templates in one scene 
               
            res = cv2.matchTemplate(output_img,template,method)
            loc = np.where( res >= CORR_MAP_RES_TH) #return 
            res = zip(*loc[::-1])      
            min_val, max_score_of_current_img, min_loc, max_loc = cv2.minMaxLoc(res)
        
        except Exception as e : #assuming only single template object in the scene 

            res = cv2.matchTemplate(output_img,template,method)

            min_val, max_score_of_current_img, min_loc, max_loc = cv2.minMaxLoc(res)
           
            if Blur: 
            
                res = cv2.blur(res,(BLUR_KER_SIZE,BLUR_KER_SIZE), 0)

                min_val, max_score_of_current_img, min_loc, max_loc = cv2.minMaxLoc(res)

        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc

        bottom_right = (top_left[0] + w, top_left[1] + h)
        
        detection_coords = [top_left,bottom_right]

        num_of_clusters = None
        if max_score_of_current_img > CORR_MAP_RES_TH:
            _ , num_of_clusters = analyze_correlation_map(res,PARAM_SEGMENTATION_BY_COLOR_TH)

        cv2.circle(res, top_left, radius=1, color=(0, 255, 0), thickness=10) #plot max_loc point
        cv2.rectangle(scene,top_left, bottom_right, 0, 30)                   #plot bbox around detection

        pair_image_template_matching_result = plots_opencv_image_pair(template,scene.copy(),show = False)    
            
    return max_score_of_current_img, detection_coords, pair_image_template_matching_result ,num_of_clusters

