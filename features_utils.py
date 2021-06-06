import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
 
def drawKeyPts(im,keyp,col,th):
    for curKey in keyp:
        x=np.int(curKey.pt[0])
        y=np.int(curKey.pt[1])
        size = np.int(curKey.size)
        size = 0 
        cv2.circle(im,(x,y),size, col,thickness=th, lineType=8, shift=0) 
    plt.imshow(im)    
    plt.plot()
    return im  
     
def Plot_img_cv2(img, str_name='_',resize_flag = False,height=400):
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
        concatenated_image = cv2.hconcat((image1, image2)) # or vconcat for vertical concatenation
        cv2.imwrite(os.path.join(output_dir_path,img_name + '.jpg'), concatenated_image)

# plots_opencv_images_pair('posters_keypoints_sift','posters_keypoints_orb','compare_feature_extractors')