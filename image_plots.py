import cv2
from cv2 import BFMatcher as bf
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from img_utils import plot_img_opencv
from skimage.transform import resize as skimage_resize 
    



def plots_opencv_images_pair_from_dir(dir1_path,dir2_path,output_dir_path = None,show = False):

    for img1,img2 in zip(sorted(glob.glob(dir1_path + '/*.jpg')),sorted(glob.glob(dir2_path + '/*.jpg'))):
        img_name = os.path.basename(img1)[:-len('.jpg')]
        image1 = cv2.imread(img1)
        image2 = cv2.imread(img2) 
        concatenated_image = cv2.hconcat((image1, image2)) 
        
        if output_dir_path != None:
            cv2.imwrite(os.path.join(output_dir_path,img_name + '.jpg'), concatenated_image)

        if show :
            plot_img_opencv(concatenated_image)



def plots_opencv_image_pair(image1,image2,show = False,convert_to_uint8 = True):
    
    """
        There is conversion to uint8 so plots only black-white images 
        conversion due to different images.dtypes 
    """

    image1, image2  = image1.copy() , image2.copy()

    if image1.shape != image2.shape : 

        if image1.size <= image2.size :
            
            try:
                
                if image2.shape[2] is not None : # HWC , rgb image
                    image1 = skimage_resize(image1.copy(), ( image2.shape[0] , image2.shape[1],image2.shape[2] ), anti_aliasing=True )
            
            except Exception as e: 

                    image1 = skimage_resize(image1.copy(), ( image2.shape[0] , image2.shape[1] ), anti_aliasing=True )
            
        else : 
            
            try:
                if image1.shape[2] is not None : 
                    image2 = skimage_resize(image2.copy(), ( image1.shape[0] , image1.shape[1],image1.shape[2] ), anti_aliasing=True )
            except Exception as e: 
                    image2 = skimage_resize(image2.copy(), ( image1.shape[0] , image1.shape[1] ), anti_aliasing=True )


    if convert_to_uint8 : 

        if image1.dtype != np.uint8:
            image1 *= 255
            image1 = image1.astype(np.uint8)

        if image2.dtype != np.uint8:
            image2 *= 255
            image2 = image2.astype(np.uint8)

    # try:
        
    concatenated_image = np.hstack([image1,image2])
    
    if show :

        # concatenated_image = cv2.cvtColor(concatenated_image,cv2.COLOR_BGR2RGB)
        plt.imshow(concatenated_image)
        plt.xticks([])
        plt.yticks([])
        plt.show()

    return concatenated_image    





        
