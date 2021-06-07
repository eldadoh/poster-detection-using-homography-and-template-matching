import glob 
import os 
import sys
import cv2
import numpy as np
from skimage.transform import resize as skimage_resize 
from skimage.metrics import structural_similarity
from features_utils import Plot_img_cv2,plots_opencv_image_pair,create_dir_with_override
from matplotlib import pyplot as plt

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
        
        Plot_img_cv2(diff,resize_flag=True)

    

# find all matches of the template in the image
# returns an array of (x, y) coordinate of the top/left point of each match
# def getMatches(image, template, threshold):
#     result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
#     loc = np.where( result >= threshold)
#     results = zip(*loc[::-1])
#     return results
    
# # Highlight regions of interest in an image
# def highlightRois(image, roisCoords, roiWidthHeight):
#     rois = []
#     for roiCoord in roisCoords:
#         roiTopLeft = roiCoord['topLeft']
#         name = roiCoord['name']
#         # extract the regions of interest from the image
#         roiBottomRight = tuple([sum(x) for x in zip(roiTopLeft, roiWidthHeight)])
#         roi = image[roiTopLeft[1]:roiBottomRight[1], roiTopLeft[0]:roiBottomRight[0]]
#         rois.append({'topLeft': roiTopLeft, 'bottomRight': roiBottomRight, 'area': roi, 'name': name})

#     # construct a darkened transparent 'layer' to darken everything
#     # in the image except for the regions of interest
#     mask = np.zeros(image.shape, dtype = "uint8")
#     image = cv2.addWeighted(image, 0.25, mask, 0.75, 0)

#     # put the original rois back in the image so that they look 'brighter'
#     for roi in rois:
#         image[roi['topLeft'][1]:roi['bottomRight'][1], roi['topLeft'][0]:roi['bottomRight'][0]] = roi['area']
#         cv2.putText(image, roi['name'][0], roi['topLeft'], cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        
#     return image

def template_matching_func(scene_path,template_path,output_path,show = False,save = False):

    """
        Input : scene image ,template 
        Returns : img ,res 
    """
    
    img_name = os.path.basename(scene_path)
    img_name = img_name[:-len('.jpg')]
    template_name = os.path.basename(template_path)

    img = cv2.imread(scene_path,0)
    img2 = img.copy()
    # _, img = cv2.threshold(img, 0, 255,cv2.THRESH_OTSU)
    template = cv2.imread(template_path,0)

    w, h = template.shape[::-1]

    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED' ] 
    # , 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

    for i,meth in enumerate(methods):
        img = img2.copy()
        method = eval(meth)
        res = cv2.matchTemplate(img,template,method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc

        
        # bottom_right = (top_left[0] + w, top_left[1] + h)

        # cv2.rectangle(img,top_left, bottom_right, 0, 30)
                
        loc = np.where(res >= 0.36) #THRESHOLD

        for pt in zip(*loc[::-1]):
                cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0,255,0), 3)
                
        if show :
            
            Plot_img_cv2(img)
            Plot_img_cv2(res)

        if save:
            path = os.path.join(output_path ,img_name + '_' + f'{template_name}' +'_' +f'{meth}' +'.jpg')
            cv2.imwrite(path,img)

    return img , res 


def template_matching_for_dir_of_images(templates_dir_path,scene_image,output_path,show = False , save = False):

    if show :

        create_dir_with_override(output_path)

    for img in glob.glob(templates_dir_path + '/*.jpg'): 

        template_matching_func(scene_image,img,output_path,show,save)
        

def main(): 

    poster_planogram_image = 'Data_new/planograms/planograms_parsed_images/APPBARBSDMN24x150421.jpg'
    templates_dir_path = 'Data_new/planograms/planograms_parsed_images'
    scene_image = '/home/arpalus/Work_Eldad/Arpalus_Code/Eldad-Local/arpalus-poster_detection/Data_new/realograms/valid_jpg_format_realograms_images/IMG_1559.jpg'
    output_path = 'Output/template_matching_output'

    #template_matching_func(scene_image,poster_planogram_image,output_path ,save=True)

    template_matching_for_dir_of_images(templates_dir_path,scene_image,output_path,show = False , save = True)

if __name__ == "__main__":

    main()