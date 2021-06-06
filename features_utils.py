import cv2
import numpy as np
import matplotlib.pyplot as plt

def drawKeyPts(im,keyp,col,th):
    for curKey in keyp:
        x=np.int(curKey.pt[0])
        y=np.int(curKey.pt[1])
        size = np.int(curKey.size)
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
