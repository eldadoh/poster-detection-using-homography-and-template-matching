import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from cv2 import BFMatcher as bf


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
     
def Plot_img_cv2(img, str_name='_',cvtcolor_flag = True ,resize_flag = False,height=400):
    if cvtcolor_flag == True :
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
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



def plots_opencv_images_pair(dir1_path,dir2_path,output_dir_path):

    for img1,img2 in zip(sorted(glob.glob(dir1_path + '/*.jpg')),sorted(glob.glob(dir2_path + '/*.jpg'))):
        img_name = os.path.basename(img1)[:-len('.jpg')]
        image1 = cv2.imread(img1)
        image2 = cv2.imread(img2) 
        concatenated_image = cv2.hconcat((image1, image2)) 
        cv2.imwrite(os.path.join(output_dir_path,img_name + '.jpg'), concatenated_image)

