import numpy as np 
import cv2 
import glob
import os
from opencv_utils import Calc_hist_grayscale
from image_plots import plot_img_opencv
from img_utils import dilate, plot_img_matplotlib,threshold_otsu,calc_image_range
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

def regionprops_criteria_for_template_matching(correlation_img_path):

    """
        gets thresholded img , and apply region props on it , to count number of clusters 
        on the thrsholded img 
    """

    try : 
        img = cv2.imread(correlation_img_path,0)
    except Exception as e : 
        img = correlation_img_path.copy()
    
    img = threshold_otsu(img)
    
    labeled_img = label(img,connectivity=2)
    
    props=regionprops(labeled_img)
    
    labeled_img = np.ascontiguousarray(labeled_img, dtype=np.uint8)
    
    all_props_list = []

    for prop in props:
        
        area = prop.area
        centroid = prop.centroid
        all_props_list.append([area,centroid])

    # print(f'{all_props_list}+\n')
    all_props_list = sorted(all_props_list, key=lambda x: x[0]) 
    # print(f'{all_props_list}+\n')
    background_ = all_props_list.pop(0)
    # print(f'{all_props_list}+\n')
    valid_centroids = [elem[1] for elem in all_props_list]
    
    for point in valid_centroids:
        y,x = int(point[0]),int(point[1])
        cv2.circle(labeled_img,(x,y),5,(1,1,1))
    
    num_of_clusters = len(valid_centroids)

    return labeled_img, num_of_clusters

def main(): 

    img_path = 'Figure_2.png'
    labeled_img, num_of_clusters  = regionprops_criteria_for_template_matching(img_path)
    # print(num_of_clusters)
    # plot_img_matplotlib(labeled_img)

if __name__ == "__main__" : 
    main() 
    # pass  