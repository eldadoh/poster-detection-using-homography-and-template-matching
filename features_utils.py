import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from cv2 import BFMatcher as bf
import shutil

def create_dir_with_override(dir_path):
    try : 
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path)
    except Exception as e : 
        print(e)
        print('Could not create the desired dir with the corersponding dir path : \n' + f'{dir_path}')


def knn_matcher(des1,des2):
    matches = bf.knnMatch(des1, des2, k=2) 
    good = []
    for (m1, m2) in matches: # for every descriptor, take closest two matches
        if m1.distance < 0.7 * m2.distance: # best match has to be this much closer than second best
            good.append(m1)

def pad_image_on_borders(img):

    return np.pad(img ,((2, 2), (2, 2),(0,0)), 'constant' , constant_values = (255))

def drawKeyPts(img,kps,col,th,circle_visualization = False):
    
    for key_point in kps:

        x=np.int(key_point.pt[0])
        y=np.int(key_point.pt[1])
        
        size = 0

        if circle_visualization :
            size = np.int(key_point.size)
        cv2.circle(img,(x,y),size, col,thickness=th, lineType=8, shift=0) 

    plt.imshow(img)    
    plt.plot()
    return img  
     
def Plot_img_cv2(img, str_name='_',cvtcolor_flag = False ,resize_flag = True,height=400):
    if resize_flag:
        img = Resize(img,height)
    cv2.imshow(str_name, img)
    k = cv2.waitKey(0)
    if k != ord('s'):
        cv2.destroyAllWindows()
    elif k == ord('s'):
        cv2.imwrite(str_name+'.jpg', img)
        cv2.destroyAllWindows()

def Resize(image, width=None, height=None, inter=cv2.INTER_AREA):

    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation=inter)

    return resized



def plots_opencv_images_pair_from_dir(dir1_path,dir2_path,output_dir_path = None,show = False):

    for img1,img2 in zip(sorted(glob.glob(dir1_path + '/*.jpg')),sorted(glob.glob(dir2_path + '/*.jpg'))):
        img_name = os.path.basename(img1)[:-len('.jpg')]
        image1 = cv2.imread(img1)
        image2 = cv2.imread(img2) 
        concatenated_image = cv2.hconcat((image1, image2)) 
        
        if output_dir_path != None:
            cv2.imwrite(os.path.join(output_dir_path,img_name + '.jpg'), concatenated_image)

        if show :
            Plot_img_cv2(concatenated_image)

def plots_opencv_image_pair(image1,image2,args,show = False):

    # concatenated_image = cv2.hconcat((image1, image2)) 
    
    if np.max(image1) <= 1 :
        image1 *= 255

    if np.max(image2) <= 1 : 
        image2 *= 255
    
    concatenated_image = np.hstack([image1,image2])

    if not args : 

        for arg in args : 

            if np.max(arg) <= 1 :
                arg *= 255 

            concatenated_image = np.hstack([concatenated_image,arg])

    if show :

            # concatenated_image = cv2.cvtColor(concatenated_image,cv2.COLOR_BGR2RGB)
            plt.imshow(concatenated_image)
            plt.show()

    return concatenated_image
