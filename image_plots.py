import cv2
from cv2 import BFMatcher as bf
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from img_utils import Resize

def plot_img_cv2(img, str_name='_',cvtcolor_flag = False ,resize_flag = True,height=400):
    if resize_flag:
        img = Resize(img,height)
    cv2.imshow(str_name, img)
    k = cv2.waitKey(0)
    if k != ord('s'):
        cv2.destroyAllWindows()
    elif k == ord('s'):
        cv2.imwrite(str_name+'.jpg', img)
        cv2.destroyAllWindows()

def plots_opencv_image_pair(image1,image2,args,show = False):
    
    if np.max(image1) <= 1 :
        image1 *= 255
        image1 = image1.astype(np.uint8)

    if np.max(image2) <= 1 : 
        image2 *= 255
        image2 = image2.astype(np.uint8)

    concatenated_image = np.hstack([image1,image2])

    if not args : 

        for arg in args : 

            if np.max(arg) <= 1 :
                arg *= 255
                arg = arg.astype(np.uint8) 

            concatenated_image = np.hstack([concatenated_image,arg])

    if show :

            # concatenated_image = cv2.cvtColor(concatenated_image,cv2.COLOR_BGR2RGB)
            plt.imshow(concatenated_image)
            plt.show()

    return concatenated_image
    

def plots_opencv_images_pair_from_dir(dir1_path,dir2_path,output_dir_path = None,show = False):

    for img1,img2 in zip(sorted(glob.glob(dir1_path + '/*.jpg')),sorted(glob.glob(dir2_path + '/*.jpg'))):
        img_name = os.path.basename(img1)[:-len('.jpg')]
        image1 = cv2.imread(img1)
        image2 = cv2.imread(img2) 
        concatenated_image = cv2.hconcat((image1, image2)) 
        
        if output_dir_path != None:
            cv2.imwrite(os.path.join(output_dir_path,img_name + '.jpg'), concatenated_image)

        if show :
            plot_img_cv2(concatenated_image)


def drawKeyPts_single_image(img,kps,col,th,circle_visualization = False,show = False):
    
    for key_point in kps:

        x=np.int(key_point.pt[0])
        y=np.int(key_point.pt[1])
        
        size = 0

        if circle_visualization :
            size = np.int(key_point.size)
        
        cv2.circle(img,(x,y),size, col,thickness=th, lineType=8, shift=0) 
   
    if show : 

        plt.imshow(img)    
        plt.plot()
    
    return img  